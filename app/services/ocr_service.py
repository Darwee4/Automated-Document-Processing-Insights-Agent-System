import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple
import uuid
from datetime import datetime
import io

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
import magic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.logging import get_agent_logger, get_cost_logger


class OCRService:
    """OCR service for text extraction from documents"""
    
    def __init__(self):
        self.logger = get_agent_logger("OCR_Agent")
        self.cost_logger = get_cost_logger()
        self.supported_formats = settings.security.allowed_extensions
        
        # Configure Tesseract
        if hasattr(settings.ocr, 'tesseract_cmd') and settings.ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.ocr.tesseract_cmd
    
    def validate_file(self, file_content: bytes, filename: str) -> Tuple[bool, str]:
        """Validate file type and size"""
        try:
            # Check file size
            max_size_bytes = settings.security.max_file_size_mb * 1024 * 1024
            if len(file_content) > max_size_bytes:
                return False, f"File size exceeds maximum allowed size of {settings.security.max_file_size_mb}MB"
            
            # Check file extension
            file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if file_ext not in self.supported_formats:
                return False, f"Unsupported file format. Allowed: {', '.join(self.supported_formats)}"
            
            # Check MIME type using python-magic
            mime_type = magic.from_buffer(file_content, mime=True)
            allowed_mime_types = [
                'application/pdf',
                'image/png', 
                'image/jpeg',
                'image/jpg',
                'image/tiff'
            ]
            
            if mime_type not in allowed_mime_types:
                return False, f"Unsupported MIME type: {mime_type}"
            
            return True, "File validation passed"
            
        except Exception as e:
            self.logger.error(f"File validation failed: {str(e)}")
            return False, f"File validation failed: {str(e)}"
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        if not settings.ocr.preprocessing_enabled:
            return image
        
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(thresh)
            
            return processed_image
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed, using original: {str(e)}")
            return image
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def extract_text_from_image(self, image: Image.Image, language: str = 'eng') -> Dict[str, Any]:
        """Extract text from a single image using Tesseract"""
        try:
            start_time = datetime.now()
            
            # Preprocess image if enabled
            if settings.ocr.preprocessing_enabled:
                image = self.preprocess_image(image)
            
            # Configure Tesseract
            config = '--oem 3 --psm 6'
            if language:
                config += f' -l {language}'
            
            # Extract text and data
            extracted_text = pytesseract.image_to_string(image, config=config)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
            
            # Calculate confidence score
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'text': extracted_text.strip(),
                'confidence': avg_confidence / 100.0,  # Convert to 0-1 scale
                'word_count': len(extracted_text.split()),
                'processing_time': processing_time,
                'language': language
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            raise
    
    def process_pdf(self, file_content: bytes, language: str = 'eng') -> Dict[str, Any]:
        """Process PDF file and extract text from all pages"""
        try:
            start_time = datetime.now()
            
            # Convert PDF to images
            images = convert_from_bytes(file_content)
            total_pages = len(images)
            
            self.logger.info(f"Processing PDF with {total_pages} pages")
            
            page_results = []
            all_text = []
            total_confidence = 0
            
            for page_num, image in enumerate(images, 1):
                self.logger.info(f"Processing page {page_num}/{total_pages}")
                
                page_result = self.extract_text_from_image(image, language)
                page_result['page_number'] = page_num
                
                page_results.append(page_result)
                all_text.append(page_result['text'])
                total_confidence += page_result['confidence']
            
            # Calculate overall metrics
            overall_confidence = total_confidence / total_pages if total_pages > 0 else 0
            full_text = '\n\n'.join(all_text)
            total_processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'extracted_text': full_text,
                'confidence_score': overall_confidence,
                'total_pages': total_pages,
                'page_results': page_results,
                'processing_time': total_processing_time,
                'ocr_engine': 'tesseract',
                'language': language
            }
            
            # Track cost
            self.cost_logger.track_ocr_usage(
                pages_processed=total_pages,
                cost=0.0,  # Tesseract is free
                document_id="pdf_processing"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"PDF processing failed: {str(e)}")
            raise
    
    def process_image(self, file_content: bytes, language: str = 'eng') -> Dict[str, Any]:
        """Process single image file"""
        try:
            start_time = datetime.now()
            
            # Open image
            image = Image.open(io.BytesIO(file_content))
            
            # Extract text
            result = self.extract_text_from_image(image, language)
            
            # Add additional metadata
            result.update({
                'total_pages': 1,
                'page_results': [{
                    'page_number': 1,
                    'text': result['text'],
                    'confidence': result['confidence'],
                    'processing_time': result['processing_time']
                }],
                'ocr_engine': 'tesseract',
                'language': language
            })
            
            # Rename key for consistency
            result['extracted_text'] = result.pop('text')
            result['confidence_score'] = result.pop('confidence')
            
            # Track cost
            self.cost_logger.track_ocr_usage(
                pages_processed=1,
                cost=0.0,  # Tesseract is free
                document_id="image_processing"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {str(e)}")
            raise
    
    async def process_document(self, file_content: bytes, filename: str, language: str = 'eng') -> Dict[str, Any]:
        """Main method to process any supported document"""
        try:
            self.logger.agent_start(document_id=filename)
            
            # Validate file
            is_valid, validation_message = self.validate_file(file_content, filename)
            if not is_valid:
                raise ValueError(validation_message)
            
            # Determine file type and process accordingly
            mime_type = magic.from_buffer(file_content, mime=True)
            
            if mime_type == 'application/pdf':
                result = self.process_pdf(file_content, language)
            else:
                result = self.process_image(file_content, language)
            
            # Add file metadata
            result.update({
                'filename': filename,
                'file_size': len(file_content),
                'mime_type': mime_type,
                'processed_at': datetime.utcnow().isoformat()
            })
            
            self.logger.agent_complete(
                document_id=filename,
                result={
                    'pages_processed': result['total_pages'],
                    'confidence_score': result['confidence_score'],
                    'processing_time': result['processing_time']
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.agent_error(
                document_id=filename,
                error=e,
                extra={'filename': filename, 'file_size': len(file_content)}
            )
            raise
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported OCR languages"""
        try:
            # Get available languages from Tesseract
            languages = pytesseract.get_languages()
            return languages
        except Exception as e:
            self.logger.error(f"Failed to get supported languages: {str(e)}")
            return ['eng']  # Default to English
    
    def get_ocr_engine_info(self) -> Dict[str, Any]:
        """Get information about the OCR engine"""
        try:
            # Get Tesseract version
            version = pytesseract.get_tesseract_version()
            
            return {
                'engine': 'tesseract',
                'version': str(version),
                'supported_languages': self.get_supported_languages(),
                'preprocessing_enabled': settings.ocr.preprocessing_enabled,
                'confidence_threshold': settings.ocr.ocr_confidence_threshold
            }
        except Exception as e:
            self.logger.error(f"Failed to get OCR engine info: {str(e)}")
            return {
                'engine': 'tesseract',
                'error': str(e)
            }


class CloudOCRService:
    """Cloud OCR service for AWS Textract or Google Vision (placeholder for future implementation)"""
    
    def __init__(self):
        self.logger = get_agent_logger("Cloud_OCR_Agent")
        self.cost_logger = get_cost_logger()
    
    async def process_with_aws_textract(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process document using AWS Textract"""
        # Placeholder for AWS Textract implementation
        # This would require AWS credentials and boto3 setup
        self.logger.info("AWS Textract processing not implemented yet")
        raise NotImplementedError("AWS Textract integration not implemented")
    
    async def process_with_google_vision(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process document using Google Vision"""
        # Placeholder for Google Vision implementation
        # This would require Google Cloud credentials
        self.logger.info("Google Vision processing not implemented yet")
        raise NotImplementedError("Google Vision integration not implemented")


# Factory function to get OCR service
def get_ocr_service(use_cloud: bool = None) -> OCRService:
    """Get appropriate OCR service based on configuration"""
    if use_cloud is None:
        use_cloud = settings.ocr.use_cloud_ocr
    
    if use_cloud:
        return CloudOCRService()
    else:
        return OCRService()


# Global OCR service instance
ocr_service = OCRService()
