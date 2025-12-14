from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import json
import os
import base64
from io import BytesIO

# PDF Libraries
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from PIL import Image as PILImage

# Flask App Initialization
app = Flask(__name__)

# Configuration Constants
AVAILABLE_MODELS = {
    'model1': {'path': 'best (1).pt', 'name': 'Model 1 (Original)'},
    'model2': {'path': 'best2.pt', 'name': 'Model 2 (Retrained)'}
}
CURRENT_MODEL = 'model1'  # Default model
REPORTS_DIR = "reports"
CAPTURES_DIR = "captures"

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)

# Global variables for detection tracking
detection_state = {
    'is_running': False,
    'session_start': None,
    'session_end': None,
    'total_detections': 0,
    'current_frame_detections': 0,
    'max_detections_in_frame': 0,
    'frame_count': 0,
    'captured_frames': [],  # Store frames with detections for report
    'last_capture_time': 0
}

class PotholeDetector:
    """YOLOv8 Pothole Detection Engine with multi-model support"""
    _instances = {}  # Store multiple model instances
    
    @classmethod
    def get_instance(cls, model_key=None):
        global CURRENT_MODEL
        if model_key is None:
            model_key = CURRENT_MODEL
        
        if model_key not in cls._instances:
            cls._instances[model_key] = cls(model_key)
        return cls._instances[model_key]
    
    @classmethod
    def switch_model(cls, model_key):
        global CURRENT_MODEL
        if model_key in AVAILABLE_MODELS:
            CURRENT_MODEL = model_key
            return cls.get_instance(model_key)
        return None
    
    def __init__(self, model_key):
        model_info = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS['model1'])
        print(f"Loading YOLOv8 model: {model_info['name']}...")
        self.model = YOLO(model_info['path'])
        self.model_name = model_info['name']
        print(f"Model loaded successfully: {model_info['name']}")
        
        self.box_annotator = sv.BoxAnnotator(
            thickness=3,
            color=sv.Color.from_hex("#FF4444")
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.8,
            text_thickness=2,
            text_color=sv.Color.WHITE,
            text_padding=10,
            color=sv.Color.from_hex("#FF4444")
        )
        
    def process_frame(self, frame):
        """Process a single frame and return annotated frame with detections"""
        global detection_state
        
        results = self.model(frame, conf=0.5)[0]  # Balanced confidence threshold
        detections = sv.Detections.from_ultralytics(results)
        
        # Update detection statistics
        num_detections = len(detections)
        detection_state['current_frame_detections'] = num_detections
        detection_state['total_detections'] += num_detections
        detection_state['frame_count'] += 1
        
        if num_detections > detection_state['max_detections_in_frame']:
            detection_state['max_detections_in_frame'] = num_detections
        
        # Create labels with confidence
        labels = []
        for i, confidence in enumerate(detections.confidence):
            labels.append(f"Lubang {confidence:.0%}")
        
        # Apply annotations
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        # Capture frames with detections for report (max 5, every 3 seconds)
        import time
        current_time = time.time()
        if num_detections > 0 and len(detection_state['captured_frames']) < 5:
            if current_time - detection_state['last_capture_time'] > 3:
                # Save frame for report
                detection_state['captured_frames'].append({
                    'frame': annotated_frame.copy(),
                    'detections': num_detections,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
                detection_state['last_capture_time'] = current_time
        
        return annotated_frame, num_detections

# Initialize detector at startup
detector = None

@app.route('/get_models')
def get_models():
    """Get available models"""
    global CURRENT_MODEL
    models = []
    for key, info in AVAILABLE_MODELS.items():
        models.append({
            'key': key,
            'name': info['name'],
            'active': key == CURRENT_MODEL
        })
    return jsonify({'models': models, 'current': CURRENT_MODEL})

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    global detector, CURRENT_MODEL
    data = request.get_json()
    model_key = data.get('model_key')
    
    if model_key not in AVAILABLE_MODELS:
        return jsonify({'error': 'Model not found'}), 400
    
    try:
        detector = PotholeDetector.switch_model(model_key)
        CURRENT_MODEL = model_key
        return jsonify({
            'success': True, 
            'current_model': AVAILABLE_MODELS[model_key]['name'],
            'model_key': model_key
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_pdf_report(report_data, captured_frames):
    """Generate a professional PDF report"""
    report_id = report_data['report_id']
    pdf_path = os.path.join(REPORTS_DIR, f"Laporan_Deteksi_{report_id}.pdf")
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                           rightMargin=2*cm, leftMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a1a2e')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#0077b6')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6
    )
    
    # Build content
    content = []
    
    # Title
    content.append(Paragraph("🛣️ LAPORAN DETEKSI LUBANG JALAN", title_style))
    content.append(Paragraph("Road Pothole Detection Report", ParagraphStyle(
        'Subtitle', parent=styles['Normal'], fontSize=12, 
        alignment=TA_CENTER, textColor=colors.gray, spaceAfter=30
    )))
    
    # Horizontal line
    content.append(Spacer(1, 10))
    
    # Session Information
    content.append(Paragraph("📋 Informasi Sesi", heading_style))
    
    session_data = [
        ['Parameter', 'Nilai'],
        ['ID Laporan', report_data['report_id']],
        ['Waktu Mulai', format_datetime(report_data['session_start'])],
        ['Waktu Selesai', format_datetime(report_data['session_end'])],
        ['Durasi Sesi', calculate_duration(report_data['session_start'], report_data['session_end'])],
        ['Model Digunakan', report_data['model_used']],
        ['Confidence Threshold', f"{report_data['confidence_threshold']*100:.0f}%"]
    ]
    
    session_table = Table(session_data, colWidths=[200, 250])
    session_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0077b6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    content.append(session_table)
    
    content.append(Spacer(1, 20))
    
    # Detection Statistics
    content.append(Paragraph("📊 Statistik Deteksi", heading_style))
    
    stats_data = [
        ['Metrik', 'Nilai', 'Keterangan'],
        ['Total Frame Diproses', str(report_data['total_frames_processed']), 'Jumlah frame yang dianalisis'],
        ['Total Deteksi', str(report_data['total_detections']), 'Akumulasi semua deteksi'],
        ['Maks. Lubang per Frame', str(report_data['max_potholes_in_single_frame']), 'Jumlah tertinggi dalam satu frame'],
        ['Estimasi Lubang Unik', str(report_data['estimated_unique_potholes']), 'Perkiraan lubang berbeda'],
        ['Rata-rata Deteksi/Frame', str(report_data['average_detections_per_frame']), 'Rata-rata per frame']
    ]
    
    stats_table = Table(stats_data, colWidths=[150, 80, 220])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e94560')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff5f5')])
    ]))
    content.append(stats_table)
    
    content.append(Spacer(1, 20))
    
    # Captured Images
    if captured_frames:
        content.append(Paragraph("📸 Gambar Hasil Deteksi", heading_style))
        content.append(Paragraph(
            f"Berikut adalah {len(captured_frames)} tangkapan layar yang menunjukkan lubang jalan terdeteksi:",
            normal_style
        ))
        content.append(Spacer(1, 10))
        
        for i, frame_data in enumerate(captured_frames):
            # Save frame to temp file
            frame = frame_data['frame']
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(frame_rgb)
            
            # Save to buffer
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=85)
            img_buffer.seek(0)
            
            # Add to PDF
            img = Image(img_buffer, width=14*cm, height=10.5*cm)
            content.append(img)
            
            # Caption
            caption = f"Gambar {i+1}: Terdeteksi {frame_data['detections']} lubang pada {frame_data['timestamp']}"
            content.append(Paragraph(caption, ParagraphStyle(
                'Caption', parent=styles['Normal'], fontSize=10,
                alignment=TA_CENTER, textColor=colors.gray, spaceAfter=20
            )))
    
    content.append(Spacer(1, 30))
    
    # Footer
    content.append(Paragraph("─" * 60, ParagraphStyle('Line', alignment=TA_CENTER)))
    content.append(Spacer(1, 10))
    content.append(Paragraph(
        f"Laporan dibuat otomatis pada {datetime.now().strftime('%d %B %Y, %H:%M:%S')}",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, 
                      alignment=TA_CENTER, textColor=colors.gray)
    ))
    content.append(Paragraph(
        "Road Pothole Detection System - YOLOv8",
        ParagraphStyle('Footer2', parent=styles['Normal'], fontSize=9, 
                      alignment=TA_CENTER, textColor=colors.gray)
    ))
    
    # Build PDF
    doc.build(content)
    
    return pdf_path

def format_datetime(iso_string):
    """Format ISO datetime to readable format"""
    if not iso_string:
        return '-'
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime('%d/%m/%Y %H:%M:%S')
    except:
        return iso_string

def calculate_duration(start, end):
    """Calculate duration between two ISO datetime strings"""
    if not start or not end:
        return '-'
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        duration = end_dt - start_dt
        total_seconds = int(duration.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes} menit {seconds} detik"
    except:
        return '-'

@app.before_request
def init_detector():
    global detector
    if detector is None:
        detector = PotholeDetector.get_instance()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Receive frame from browser camera and return detection result"""
    global detector, detection_state
    
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode base64 image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Process frame with detection
        annotated_frame, num_detections = detector.process_frame(frame)
        
        # Encode result back to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{result_base64}',
            'detections': num_detections,
            'total_detections': detection_state['total_detections'],
            'max_in_frame': detection_state['max_detections_in_frame'],
            'frame_count': detection_state['frame_count']
        })
        
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/start_session', methods=['POST'])
def start_session():
    """Start a new detection session"""
    global detection_state
    
    detection_state = {
        'is_running': True,
        'session_start': datetime.now().isoformat(),
        'session_end': None,
        'total_detections': 0,
        'current_frame_detections': 0,
        'max_detections_in_frame': 0,
        'frame_count': 0,
        'captured_frames': [],
        'last_capture_time': 0
    }
    
    return jsonify({'success': True, 'session_start': detection_state['session_start']})

@app.route('/stop_session', methods=['POST'])
def stop_session():
    """Stop detection session and generate report"""
    global detection_state
    
    detection_state['is_running'] = False
    detection_state['session_end'] = datetime.now().isoformat()
    
    # Calculate statistics
    avg_detections = 0
    if detection_state['frame_count'] > 0:
        avg_detections = detection_state['total_detections'] / detection_state['frame_count']
    
    # Generate report data
    report = {
        'report_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'session_start': detection_state['session_start'],
        'session_end': detection_state['session_end'],
        'total_frames_processed': detection_state['frame_count'],
        'total_detections': detection_state['total_detections'],
        'max_potholes_in_single_frame': detection_state['max_detections_in_frame'],
        'estimated_unique_potholes': detection_state['max_detections_in_frame'],
        'average_detections_per_frame': round(avg_detections, 2),
        'model_used': 'YOLOv8',
        'confidence_threshold': 0.75
    }
    
    # Save JSON report
    report_filename = f"report_{report['report_id']}.json"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate PDF report
    pdf_path = None
    try:
        pdf_path = generate_pdf_report(report, detection_state['captured_frames'])
        report['pdf_file'] = os.path.basename(pdf_path)
    except Exception as e:
        print(f"Error generating PDF: {e}")
        report['pdf_file'] = None
    
    return jsonify({
        'success': True,
        'report': report,
        'report_file': report_filename,
        'pdf_file': report.get('pdf_file')
    })

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    """Download PDF report"""
    pdf_path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True, download_name=filename)
    return jsonify({'error': 'File not found'}), 404

@app.route('/detection_stats')
def get_detection_stats():
    """Get current detection statistics"""
    return jsonify({
        'is_running': detection_state['is_running'],
        'current_detections': detection_state['current_frame_detections'],
        'total_detections': detection_state['total_detections'],
        'max_in_frame': detection_state['max_detections_in_frame'],
        'frame_count': detection_state['frame_count'],
        'session_start': detection_state['session_start'],
        'captured_count': len(detection_state.get('captured_frames', []))
    })

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Process uploaded image for pothole detection"""
    global detector
    
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode base64 image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Run detection
        results = detector.model(frame, conf=0.5)[0]
        detections = sv.Detections.from_ultralytics(results)
        num_detections = len(detections)
        
        # Create labels
        labels = [f"Lubang {conf:.0%}" for conf in detections.confidence]
        
        # Annotate frame
        annotated = detector.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = detector.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Generate PDF report for the image
        pdf_filename = None
        report_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f"Laporan_Foto_{report_id}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        
        # Create comprehensive PDF
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        content = []
        
        # Title
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, alignment=TA_CENTER, 
                                      textColor=colors.HexColor('#1a1a2e'), spaceAfter=10)
        content.append(Paragraph("📷 LAPORAN ANALISIS FOTO", title_style))
        content.append(Paragraph("Photo Analysis Report - Deteksi Lubang Jalan", ParagraphStyle(
            'Subtitle', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER, 
            textColor=colors.gray, spaceAfter=20)))
        content.append(Spacer(1, 10))
        
        # Section 1: Info Dokumen
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=13, 
                                        textColor=colors.HexColor('#e94560'), spaceBefore=15, spaceAfter=8)
        content.append(Paragraph("📋 Informasi Dokumen", heading_style))
        
        doc_info = [
            ['Parameter', 'Nilai'],
            ['ID Laporan', f'FOTO-{report_id}'],
            ['Tanggal Analisis', datetime.now().strftime('%d %B %Y')],
            ['Waktu Analisis', datetime.now().strftime('%H:%M:%S WIB')],
            ['Tipe Sumber', 'Upload Foto'],
            ['Model AI', f'YOLOv8 - {AVAILABLE_MODELS.get(CURRENT_MODEL, {}).get("name", "Model 1")}'],
        ]
        
        table = Table(doc_info, colWidths=[180, 270])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e94560')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fff5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ffccd5')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        content.append(table)
        content.append(Spacer(1, 15))
        
        # Section 2: Hasil Deteksi
        content.append(Paragraph("🔍 Hasil Deteksi", heading_style))
        
        detection_info = [
            ['Metrik', 'Hasil'],
            ['Total Lubang Terdeteksi', str(num_detections)],
            ['Status', '⚠️ DITEMUKAN LUBANG' if num_detections > 0 else '✅ AMAN'],
            ['Confidence Threshold', '50%'],
            ['Tingkat Keyakinan', ', '.join([f'{c:.0%}' for c in detections.confidence]) if num_detections > 0 else '-'],
        ]
        
        table2 = Table(detection_info, colWidths=[180, 270])
        status_color = colors.HexColor('#fff3cd') if num_detections > 0 else colors.HexColor('#d4edda')
        table2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0077b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f9ff')),
            ('BACKGROUND', (1, 2), (1, 2), status_color),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#b8daff')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        content.append(table2)
        content.append(Spacer(1, 20))
        
        # Section 3: Gambar Hasil Analisis
        content.append(Paragraph("📸 Gambar Hasil Analisis", heading_style))
        content.append(Paragraph("Gambar berikut menunjukkan hasil deteksi dengan bounding box merah pada area yang teridentifikasi sebagai lubang jalan.", 
            ParagraphStyle('Note', parent=styles['Normal'], fontSize=9, textColor=colors.gray, spaceAfter=10)))
        
        img_buffer = BytesIO(cv2.imencode('.jpg', annotated)[1])
        rl_img = RLImage(img_buffer, width=16*cm, height=12*cm)
        content.append(rl_img)
        content.append(Spacer(1, 15))
        
        # Section 4: Rekomendasi (if potholes found)
        if num_detections > 0:
            content.append(Paragraph("⚡ Rekomendasi", heading_style))
            recommendations = [
                "• Segera lakukan perbaikan pada area yang terdeteksi",
                "• Pasang tanda peringatan untuk pengendara",
                "• Dokumentasikan lokasi untuk pelaporan ke dinas terkait",
                "• Lakukan inspeksi berkala pada area sekitar"
            ]
            for rec in recommendations:
                content.append(Paragraph(rec, ParagraphStyle('Rec', parent=styles['Normal'], fontSize=10, spaceAfter=5)))
        
        # Footer
        content.append(Spacer(1, 30))
        content.append(Paragraph("─" * 60, ParagraphStyle('Line', alignment=TA_CENTER)))
        content.append(Paragraph("Laporan ini dihasilkan secara otomatis oleh sistem Pothole Detection berbasis YOLOv8",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.gray)))
        
        doc.build(content)
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{result_base64}',
            'detections': num_detections,
            'details': [{'confidence': float(c)} for c in detections.confidence],
            'pdf_file': pdf_filename
        })
        
    except Exception as e:
        print(f"Upload image error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Process uploaded video for pothole detection"""
    global detector
    
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporarily
        temp_path = os.path.join('captures', f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        video_file.save(temp_path)
        
        # Process video
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            os.remove(temp_path)
            return jsonify({'error': 'Could not open video'}), 400
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results_data = {
            'total_frames': total_frames,
            'fps': fps,
            'frames_processed': 0,
            'total_detections': 0,
            'max_in_frame': 0,
            'frames_with_detections': 0,
            'sample_frames': []
        }
        
        frame_skip = max(1, int(fps / 2))  # Process 2 frames per second
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # Run detection
                model_results = detector.model(frame, conf=0.5)[0]
                detections = sv.Detections.from_ultralytics(model_results)
                num_detections = len(detections)
                
                results_data['frames_processed'] += 1
                results_data['total_detections'] += num_detections
                
                if num_detections > results_data['max_in_frame']:
                    results_data['max_in_frame'] = num_detections
                
                if num_detections > 0:
                    results_data['frames_with_detections'] += 1
                    
                    # Save sample frames (max 5)
                    if len(results_data['sample_frames']) < 5:
                        labels = [f"Lubang {conf:.0%}" for conf in detections.confidence]
                        annotated = detector.box_annotator.annotate(scene=frame.copy(), detections=detections)
                        annotated = detector.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                        
                        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        results_data['sample_frames'].append({
                            'frame_number': frame_count,
                            'time': f"{frame_count/fps:.1f}s",
                            'detections': num_detections,
                            'image': f'data:image/jpeg;base64,{frame_base64}'
                        })
            
            frame_count += 1
        
        cap.release()
        os.remove(temp_path)  # Clean up
        
        # Generate PDF report for video (always)
        report_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f"Laporan_Video_{report_id}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        content = []
        
        # Title
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, alignment=TA_CENTER, 
                                      textColor=colors.HexColor('#1a1a2e'), spaceAfter=10)
        content.append(Paragraph("🎬 LAPORAN ANALISIS VIDEO", title_style))
        content.append(Paragraph("Video Analysis Report - Deteksi Lubang Jalan", ParagraphStyle(
            'Subtitle', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER, 
            textColor=colors.gray, spaceAfter=20)))
        content.append(Spacer(1, 10))
        
        # Section 1: Info Dokumen
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=13, 
                                        textColor=colors.HexColor('#0077b6'), spaceBefore=15, spaceAfter=8)
        content.append(Paragraph("📋 Informasi Dokumen", heading_style))
        
        duration_sec = results_data['total_frames'] / fps if fps > 0 else 0
        doc_info = [
            ['Parameter', 'Nilai'],
            ['ID Laporan', f'VIDEO-{report_id}'],
            ['Tanggal Analisis', datetime.now().strftime('%d %B %Y')],
            ['Waktu Analisis', datetime.now().strftime('%H:%M:%S WIB')],
            ['Tipe Sumber', 'Upload Video'],
            ['Durasi Video', f'{duration_sec:.1f} detik'],
            ['FPS Video', f'{fps:.1f}'],
            ['Model AI', f'YOLOv8 - {AVAILABLE_MODELS.get(CURRENT_MODEL, {}).get("name", "Model 1")}'],
        ]
        
        table = Table(doc_info, colWidths=[180, 270])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0077b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f9ff')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#b8daff')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        content.append(table)
        content.append(Spacer(1, 15))
        
        # Section 2: Statistik Deteksi
        content.append(Paragraph("📊 Statistik Deteksi", heading_style))
        
        detection_rate = (results_data['frames_with_detections'] / results_data['frames_processed'] * 100) if results_data['frames_processed'] > 0 else 0
        
        detection_info = [
            ['Metrik', 'Hasil'],
            ['Total Frame Diproses', str(results_data['frames_processed'])],
            ['Total Lubang Terdeteksi', str(results_data['total_detections'])],
            ['Maks Lubang per Frame', str(results_data['max_in_frame'])],
            ['Frame dengan Lubang', str(results_data['frames_with_detections'])],
            ['Tingkat Deteksi', f'{detection_rate:.1f}%'],
            ['Status Jalan', '⚠️ PERLU PERHATIAN' if results_data['total_detections'] > 0 else '✅ KONDISI BAIK'],
        ]
        
        table2 = Table(detection_info, colWidths=[180, 270])
        status_color = colors.HexColor('#fff3cd') if results_data['total_detections'] > 0 else colors.HexColor('#d4edda')
        table2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('BACKGROUND', (1, 6), (1, 6), status_color),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        content.append(table2)
        content.append(Spacer(1, 20))
        
        # Section 3: Sample Frames
        if results_data['sample_frames']:
            content.append(Paragraph("📸 Bukti Deteksi", heading_style))
            content.append(Paragraph("Berikut adalah sample frame yang menunjukkan lokasi lubang jalan terdeteksi:", 
                ParagraphStyle('Note', parent=styles['Normal'], fontSize=9, textColor=colors.gray, spaceAfter=10)))
            
            for i, sf in enumerate(results_data['sample_frames'][:3]):
                img_data = sf['image'].split('base64,')[1]
                img_bytes = base64.b64decode(img_data)
                img_buffer = BytesIO(img_bytes)
                rl_img = RLImage(img_buffer, width=14*cm, height=10*cm)
                content.append(rl_img)
                content.append(Paragraph(f"Frame #{i+1} - Waktu: {sf['time']} | Lubang: {sf['detections']}", 
                    ParagraphStyle('Caption', fontSize=9, alignment=TA_CENTER, textColor=colors.gray, spaceAfter=15)))
        
        # Section 4: Rekomendasi
        if results_data['total_detections'] > 0:
            content.append(Paragraph("⚡ Rekomendasi Tindakan", heading_style))
            recommendations = [
                "• Prioritaskan perbaikan pada lokasi dengan banyak lubang terdeteksi",
                "• Gunakan data waktu frame untuk mengidentifikasi lokasi spesifik",
                "• Lakukan survei lapangan untuk verifikasi hasil deteksi",
                "• Dokumentasikan kondisi untuk evaluasi berkala",
                "• Koordinasikan dengan dinas terkait untuk penanganan"
            ]
            for rec in recommendations:
                content.append(Paragraph(rec, ParagraphStyle('Rec', parent=styles['Normal'], fontSize=10, spaceAfter=5)))
        
        # Footer
        content.append(Spacer(1, 30))
        content.append(Paragraph("─" * 60, ParagraphStyle('Line', alignment=TA_CENTER)))
        content.append(Paragraph("Laporan ini dihasilkan secara otomatis oleh sistem Pothole Detection berbasis YOLOv8",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.gray)))
        
        doc.build(content)
        
        results_data['pdf_file'] = pdf_filename
        
        return jsonify({
            'success': True,
            'results': results_data
        })
        
    except Exception as e:
        print(f"Upload video error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    # Pre-load the model
    detector = PotholeDetector.get_instance()
    # Use PORT environment variable for Render, default to 5000 for local
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
else:
    # For gunicorn/production - pre-load model
    detector = PotholeDetector.get_instance()