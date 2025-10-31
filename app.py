import os
import tempfile
import datetime
import json
import math
import threading
import time
from flask import Flask, request, render_template, send_file, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAI
import uuid
from dotenv import load_dotenv
try:
    from pydub import AudioSegment
    import ffmpeg
    AUDIO_PROCESSING_AVAILABLE = True
    print("✅ Audio processing libraries loaded successfully")
except ImportError as e:
    AUDIO_PROCESSING_AVAILABLE = False
    print(f"⚠️  Audio processing libraries not available: {e}")
    print("   Using alternative chunking strategy")

# Alternative chunking without ffmpeg
import shutil

# Global dictionary to store processing status
processing_status = {}

# Load environment variables from .env file
load_dotenv()

# Debug: Print current working directory and .env file path
print(f"🔍 Current working directory: {os.getcwd()}")
env_file_path = os.path.join(os.getcwd(), '.env')
print(f"🔍 Looking for .env file at: {env_file_path}")
print(f"🔍 .env file exists: {os.path.exists(env_file_path)}")

# Force reload the .env file with explicit path
load_dotenv(env_file_path, override=True)

# Explicitly set the API key from .env if not already in environment
if not os.getenv('OPENAI_API_KEY'):
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r') as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    api_key_value = line.split('=', 1)[1].strip()
                    os.environ['OPENAI_API_KEY'] = api_key_value
                    print(f"🔑 Manually set API key from .env file")
                    break

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRANSCRIPTS_FOLDER'] = 'transcripts'

def update_status(file_id, status, progress=0, message=""):
    """Update processing status for a file"""
    processing_status[file_id] = {
        'status': status,
        'progress': progress,
        'message': message,
        'timestamp': datetime.datetime.now().isoformat()
    }

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRANSCRIPTS_FOLDER'], exist_ok=True)

client = None

def get_openai_client():
    """Return API key for direct HTTP requests to OpenAI API."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key loaded successfully")
        return api_key.strip()
    else:
        print("⚠️ OpenAI API key not found.")
        return None

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'mp4'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_from_mp4(input_path, file_id=None):
    """Extract audio from MP4 video file and convert to MP3 format"""
    try:
        if file_id:
            update_status(file_id, 'converting', 15, 'Extracting audio from MP4 video...')
        
        print(f"🔄 Extracting audio from MP4: {os.path.basename(input_path)}")
        
        # Generate MP3 output path
        base_name = os.path.splitext(input_path)[0]
        mp3_path = f"{base_name}_audio.mp3"
        
        # Try pydub with different approaches
        try:
            # Method 1: Direct extraction with pydub
            if AUDIO_PROCESSING_AVAILABLE:
                try:
                    audio = AudioSegment.from_file(input_path, format="mp4")
                    audio.export(mp3_path, format="mp3", bitrate="128k")
                    print("✅ Extracted audio using pydub")
                except Exception as e:
                    print(f"⚠️ Pydub extraction failed: {e}")
                    raise e
            else:
                raise Exception("Audio processing libraries not available")
            
            # Verify the MP3 file was created
            if os.path.exists(mp3_path):
                mp3_size_mb = get_file_size_mb(mp3_path)
                print(f"✅ MP4 audio extracted to MP3: {os.path.basename(mp3_path)} ({mp3_size_mb:.1f}MB)")
                
                if file_id:
                    update_status(file_id, 'converted', 20, f'Successfully extracted audio to MP3 ({mp3_size_mb:.1f}MB)')
                
                # Remove original MP4 file to save space
                try:
                    os.remove(input_path)
                    print(f"🗑️ Removed original MP4 file: {os.path.basename(input_path)}")
                except Exception as e:
                    print(f"⚠️ Could not remove original MP4 file: {e}")
                
                return mp3_path
            else:
                raise Exception("MP3 extraction failed - output file not created")
                
        except Exception as conversion_error:
            print(f"❌ MP4 extraction failed: {conversion_error}")
            if file_id:
                update_status(file_id, 'error', 0, f'MP4 extraction failed: {conversion_error}')
            raise Exception(f"MP4 audio extraction is required but failed: {conversion_error}")
            
    except Exception as e:
        if file_id:
            update_status(file_id, 'error', 0, f'MP4 extraction failed: {str(e)}')
        print(f"❌ MP4 extraction failed: {e}")
        raise Exception(f"MP4 audio extraction is required but failed: {str(e)}")

def convert_m4a_to_mp3(input_path, file_id=None):
    """Convert M4A file to MP3 format for better compatibility"""
    try:
        if file_id:
            update_status(file_id, 'converting', 15, 'Converting M4A to MP3 format...')
        
        print(f"🔄 Converting M4A to MP3: {os.path.basename(input_path)}")
        
        # Generate MP3 output path
        base_name = os.path.splitext(input_path)[0]
        mp3_path = f"{base_name}_converted.mp3"
        
        # Try pydub with different approaches
        try:
            # Load M4A file - try different methods
            audio = None
            
            # Method 1: Direct format specification
            try:
                audio = AudioSegment.from_file(input_path, format="m4a")
                print("✅ Loaded M4A using direct format")
            except:
                pass
            
            # Method 2: Let pydub auto-detect
            if audio is None:
                try:
                    audio = AudioSegment.from_file(input_path)
                    print("✅ Loaded M4A using auto-detection")
                except:
                    pass
            
            # Method 3: Try as MP4 (M4A is essentially MP4)
            if audio is None:
                try:
                    audio = AudioSegment.from_file(input_path, format="mp4")
                    print("✅ Loaded M4A as MP4 format")
                except:
                    pass
            
            if audio is not None:
                # Export as MP3 with good quality settings
                audio.export(mp3_path, format="mp3", bitrate="128k")
                
                # Verify the MP3 file was created
                if os.path.exists(mp3_path):
                    mp3_size_mb = get_file_size_mb(mp3_path)
                    print(f"✅ M4A converted to MP3: {os.path.basename(mp3_path)} ({mp3_size_mb:.1f}MB)")
                    
                    if file_id:
                        update_status(file_id, 'converted', 20, f'Successfully converted to MP3 ({mp3_size_mb:.1f}MB)')
                    
                    # Remove original M4A file to save space
                    try:
                        os.remove(input_path)
                        print(f"🗑️  Removed original M4A file: {os.path.basename(input_path)}")
                    except Exception as e:
                        print(f"⚠️  Could not remove original M4A file: {e}")
                    
                    return mp3_path
                else:
                    raise Exception("MP3 conversion failed - output file not created")
            else:
                raise Exception("Could not load M4A file with any method")
                
        except Exception as conversion_error:
            print(f"❌ M4A conversion failed: {conversion_error}")
            if file_id:
                update_status(file_id, 'error', 0, f'M4A conversion failed: {conversion_error}')
            raise Exception(f"M4A conversion is required but failed: {conversion_error}")
            
    except Exception as e:
        if file_id:
            update_status(file_id, 'error', 0, f'M4A conversion failed: {str(e)}')
        print(f"❌ M4A conversion failed: {e}")
        raise Exception(f"M4A conversion is required but failed: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/upload', methods=['POST'])
def upload_file():
    if get_openai_client() is None:
        return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        saved_filename = f"{file_id}.{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        
        # Save uploaded file
        file.save(filepath)
        
        # Initialize status
        update_status(file_id, 'uploading', 10, f'File "{filename}" uploaded successfully')
        
        # Start transcription in background thread
        def background_transcription():
            try:
                # Convert M4A to MP3 or extract audio from MP4 if needed
                processing_filepath = filepath
                if file_extension.lower() == 'm4a':
                    print(f"🔄 M4A file detected, converting to MP3...")
                    processing_filepath = convert_m4a_to_mp3(filepath, file_id)
                elif file_extension.lower() == 'mp4':
                    print(f"🔄 MP4 video file detected, extracting audio...")
                    processing_filepath = extract_audio_from_mp4(filepath, file_id)
                
                transcript_filename = transcribe_audio(processing_filepath, file_id)
                update_status(file_id, 'completed', 100, 'Transcription completed successfully')
                
                # Clean up uploaded file
                if os.path.exists(processing_filepath):
                    os.remove(processing_filepath)
                    
            except Exception as e:
                update_status(file_id, 'error', 0, f'Transcription failed: {str(e)}')
                # Clean up uploaded file on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                # Also clean up converted file if it exists
                if file_extension.lower() == 'm4a':
                    converted_path = f"{os.path.splitext(filepath)[0]}_converted.mp3"
                    if os.path.exists(converted_path):
                        os.remove(converted_path)
                elif file_extension.lower() == 'mp4':
                    audio_path = f"{os.path.splitext(filepath)[0]}_audio.mp3"
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
        
        # Start background processing
        thread = threading.Thread(target=background_transcription)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'transcript_id': file_id,
            'status_url': url_for('get_processing_status', file_id=file_id)
        })
    
    return jsonify({'error': 'Invalid file type. Please upload MP3, WAV, M4A, FLAC, or OGG files.'}), 400

@app.route('/status/<file_id>')
def get_processing_status(file_id):
    """Get processing status for a file"""
    if file_id in processing_status:
        status = processing_status[file_id]
        
        # Add download URL if completed
        if status['status'] == 'completed':
            status['download_url'] = url_for('download_transcript', transcript_id=file_id)
        
        return jsonify(status)
    else:
        return jsonify({'status': 'not_found', 'message': 'File not found'}), 404

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def split_audio_file_by_bytes(filepath, target_size_mb=10, file_id=None):
    """Alternative chunking method using byte-level splitting (no ffmpeg required)"""
    try:
        file_size_mb = get_file_size_mb(filepath)
        print(f"🔍 Alternative chunking: {file_size_mb:.2f}MB file, target {target_size_mb}MB chunks")
        
        if file_id:
            update_status(file_id, 'analyzing', 15, f'Using alternative chunking method for {file_size_mb:.1f}MB file')
        
        # If file is small enough, return as-is
        if file_size_mb <= 20:
            print(f"📝 File is small enough ({file_size_mb:.1f}MB <= 20MB) - processing directly")
            if file_id:
                update_status(file_id, 'processing', 20, f'File size: {file_size_mb:.1f}MB - Processing directly')
            return [filepath]
        
        # Calculate number of chunks needed
        num_chunks = math.ceil(file_size_mb / target_size_mb)
        print(f"🔢 Calculated {num_chunks} chunks needed for {file_size_mb:.1f}MB file")
        
        # Create chunks folder
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        chunks_folder = os.path.join(os.path.dirname(filepath), f"{base_filename}_chunks")
        os.makedirs(chunks_folder, exist_ok=True)
        print(f"📁 Created chunks folder: {chunks_folder}")
        
        if file_id:
            update_status(file_id, 'splitting', 25, f'Creating {num_chunks} chunks using byte-level splitting')
        
        # Get file size in bytes
        file_size_bytes = os.path.getsize(filepath)
        chunk_size_bytes = file_size_bytes // num_chunks
        
        chunk_files = []
        
        print(f"🔍 Opening source file: {filepath}")
        print(f"📊 File size: {file_size_bytes} bytes, Chunk size: {chunk_size_bytes} bytes")
        
        with open(filepath, 'rb') as source_file:
            for i in range(num_chunks):
                try:
                    # Use original file extension for chunks
                    original_ext = os.path.splitext(filepath)[1]
                    chunk_filename = os.path.join(chunks_folder, f"chunk_{i+1:03d}{original_ext}")
                    print(f"🔨 Creating chunk {i+1}/{num_chunks} at: {chunk_filename}")
                    
                    # Calculate chunk boundaries
                    start_byte = i * chunk_size_bytes
                    if i == num_chunks - 1:  # Last chunk gets remaining bytes
                        chunk_bytes = file_size_bytes - start_byte
                    else:
                        chunk_bytes = chunk_size_bytes
                    
                    print(f"📏 Chunk {i+1}: bytes {start_byte} to {start_byte + chunk_bytes} ({chunk_bytes} bytes)")
                    
                    if file_id:
                        progress = 25 + (i / num_chunks) * 15  # Progress from 25% to 40%
                        update_status(file_id, 'splitting', progress, f'Creating chunk {i+1}/{num_chunks} ({chunk_bytes/1024/1024:.1f}MB)')
                    
                    # Copy chunk data
                    source_file.seek(start_byte)
                    chunk_data = source_file.read(chunk_bytes)
                    print(f"📖 Read {len(chunk_data)} bytes from source file")
                    
                    if len(chunk_data) == 0:
                        print(f"⚠️  Warning: No data read for chunk {i+1}")
                        continue
                    
                    with open(chunk_filename, 'wb') as chunk_file:
                        chunk_file.write(chunk_data)
                    
                    # Verify chunk file was created
                    if os.path.exists(chunk_filename):
                        chunk_files.append(chunk_filename)
                        actual_size = get_file_size_mb(chunk_filename)
                        print(f"✅ Created chunk {i+1}/{num_chunks}: {os.path.basename(chunk_filename)} ({actual_size:.1f}MB)")
                    else:
                        print(f"❌ Failed to create chunk file: {chunk_filename}")
                        
                except Exception as chunk_error:
                    print(f"❌ Error creating chunk {i+1}: {chunk_error}")
                    continue
        
        print(f"✅ Successfully created {len(chunk_files)} chunks using byte-level splitting")
        
        if file_id:
            update_status(file_id, 'ready_transcribe', 40, f'Successfully created {len(chunk_files)} chunks - Ready for sequential processing')
        
        return chunk_files
        
    except Exception as e:
        if file_id:
            update_status(file_id, 'error', 0, f'Alternative chunking failed: {str(e)}')
        print(f"⚠️  Alternative chunking failed: {e}")
        print(f"🔧 Error details: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        print(f"🔧 Falling back to direct processing")
        return [filepath]

def split_audio_file_by_size(filepath, target_size_mb=10, file_id=None):
    """Split audio file into chunks based on target file size (MB)"""
    if not AUDIO_PROCESSING_AVAILABLE:
        print(f"⚠️  Audio processing not available - using alternative byte-level chunking")
        return split_audio_file_by_bytes(filepath, target_size_mb, file_id)
    
    # Original pydub-based chunking logic (if ffmpeg is available)
    
    try:
        if file_id:
            update_status(file_id, 'analyzing', 15, 'Analyzing audio file...')
        
        # Get file size first
        file_size_mb = get_file_size_mb(filepath)
        print(f"🔍 File size check: {file_size_mb:.2f}MB (threshold: 20MB)")
        
        # If file is small enough, return as-is
        if file_size_mb <= 20:  # New threshold: 20MB
            print(f"📝 File is small enough ({file_size_mb:.1f}MB <= 20MB) - processing directly")
            if file_id:
                update_status(file_id, 'processing', 20, f'File size: {file_size_mb:.1f}MB - Processing directly')
            return [filepath]
        
        print(f"📂 File is large ({file_size_mb:.1f}MB > 20MB) - splitting required")
        
        # Load audio file
        print(f"🎵 Loading audio file for analysis...")
        audio = AudioSegment.from_file(filepath)
        print(f"✅ Audio loaded successfully - Duration: {len(audio)/1000:.1f} seconds")
        
        # Calculate chunk duration based on target file size
        total_duration = len(audio)  # in milliseconds
        duration_minutes = total_duration / (1000 * 60)  # Convert to minutes
        
        print(f"📊 Audio analysis: {duration_minutes:.1f} minutes, {file_size_mb:.1f}MB")
        
        # Estimate chunk duration to achieve target file size
        size_ratio = target_size_mb / file_size_mb
        chunk_duration_ms = int(total_duration * size_ratio)
        
        print(f"🧮 Calculated chunk duration: {chunk_duration_ms/1000:.1f} seconds (ratio: {size_ratio:.2f})")
        
        # Ensure minimum chunk duration (2 minutes) and maximum (15 minutes)
        chunk_duration_ms = max(120000, min(chunk_duration_ms, 900000))
        
        print(f"📏 Adjusted chunk duration: {chunk_duration_ms/1000:.1f} seconds")
        
        num_chunks = math.ceil(total_duration / chunk_duration_ms)
        
        print(f"🔢 Calculated number of chunks: {num_chunks}")
        
        if file_id:
            update_status(file_id, 'splitting', 25, f'Large file detected ({file_size_mb:.1f}MB, {duration_minutes:.1f} min) - Splitting into {num_chunks} chunks (~{target_size_mb}MB each)')
        
        chunk_files = []
        base_name = os.path.splitext(filepath)[0]
        chunks_folder = f"{base_name}_chunks"
        
        # Create chunks folder
        os.makedirs(chunks_folder, exist_ok=True)
        
        print(f"📂 Splitting audio file into {num_chunks} chunks of ~{target_size_mb}MB each...")
        
        # Validate that we actually have multiple chunks
        if num_chunks <= 1:
            print(f"⚠️  Warning: Calculated only {num_chunks} chunk(s) - this seems incorrect for a {file_size_mb:.1f}MB file")
            # Force at least 2 chunks for files > 20MB
            num_chunks = max(2, math.ceil(file_size_mb / target_size_mb))
            chunk_duration_ms = total_duration // num_chunks
            print(f"🔧 Forced chunk count to {num_chunks} with duration {chunk_duration_ms/1000:.1f}s each")
        
        for i in range(num_chunks):
            start_time = i * chunk_duration_ms
            end_time = min((i + 1) * chunk_duration_ms, total_duration)
            
            print(f"🔨 Creating chunk {i+1}/{num_chunks}: {start_time/1000:.1f}s - {end_time/1000:.1f}s")
            
            chunk = audio[start_time:end_time]
            # Use original file extension and format for chunks
            original_ext = os.path.splitext(filepath)[1].lower()
            if original_ext == '.m4a':
                chunk_filename = os.path.join(chunks_folder, f"chunk_{i+1:03d}.m4a")
                export_format = "mp4"  # M4A uses MP4 container
            elif original_ext == '.wav':
                chunk_filename = os.path.join(chunks_folder, f"chunk_{i+1:03d}.wav")
                export_format = "wav"
            elif original_ext == '.flac':
                chunk_filename = os.path.join(chunks_folder, f"chunk_{i+1:03d}.flac")
                export_format = "flac"
            else:
                chunk_filename = os.path.join(chunks_folder, f"chunk_{i+1:03d}.mp3")
                export_format = "mp3"
            
            if file_id:
                progress = 25 + (i / num_chunks) * 15  # Progress from 25% to 40%
                start_min, start_sec = divmod(int(start_time/1000), 60)
                end_min, end_sec = divmod(int(end_time/1000), 60)
                update_status(file_id, 'splitting', progress, f'Creating chunk {i+1}/{num_chunks} ({start_min}:{start_sec:02d} - {end_min}:{end_sec:02d})')
            
            # Export chunk in original format with optimized settings
            if export_format == "mp3":
                chunk.export(chunk_filename, format="mp3", bitrate="96k", parameters=["-ac", "1"])
            else:
                chunk.export(chunk_filename, format=export_format)
            chunk_files.append(chunk_filename)
            
            actual_size = get_file_size_mb(chunk_filename)
            print(f"📄 Created chunk {i+1}/{num_chunks}: {os.path.basename(chunk_filename)} ({actual_size:.1f}MB)")
        
        print(f"✅ Successfully created {len(chunk_files)} chunks in {chunks_folder}")
        
        if file_id:
            update_status(file_id, 'ready_transcribe', 40, f'Successfully created {len(chunk_files)} chunks in {chunks_folder} - Ready for sequential processing')
        
        # Verify we have multiple chunks
        if len(chunk_files) <= 1:
            print(f"⚠️  ERROR: Expected multiple chunks but only got {len(chunk_files)}")
        
        return chunk_files
        
    except Exception as e:
        # Check if this is an M4A file - don't use byte-level chunking as it corrupts the file
        file_ext = os.path.splitext(filepath)[1].lower()
        if file_ext == '.m4a':
            if file_id:
                update_status(file_id, 'error', 0, f'M4A chunking failed and byte-level chunking would corrupt the file. Please convert to MP3 first: {str(e)}')
            print(f"❌ M4A chunking failed: {e}")
            print(f"🚫 Cannot use byte-level chunking on M4A files - it corrupts the audio structure")
            raise Exception(f"M4A file chunking failed. Please convert to MP3 format first. Error: {str(e)}")
        
        if file_id:
            update_status(file_id, 'processing', 25, f'Audio splitting failed, using alternative method: {str(e)}')
        print(f"⚠️  Audio splitting failed: {e}")
        print(f"🔧 Falling back to byte-level chunking")
        return split_audio_file_by_bytes(filepath, target_size_mb, file_id)

def transcribe_and_append_chunk(chunk_filepath, chunk_index, time_offset, transcript_path, file_id=None, total_chunks=1):
    """Transcribe a single audio chunk and append results to transcript file"""
    try:
        if file_id:
            chunk_size_mb = get_file_size_mb(chunk_filepath)
            update_status(file_id, 'transcribing', 40 + (chunk_index / total_chunks) * 50, 
                         f'Processing chunk {chunk_index + 1}/{total_chunks} ({chunk_size_mb:.1f}MB) - Uploading and transcribing...')
        
        print(f"🎵 Transcribing chunk {chunk_index + 1}/{total_chunks}: {os.path.basename(chunk_filepath)}")
        
        with open(chunk_filepath, "rb") as audio_file:
            api_key = get_openai_client()
            if not api_key:
                raise Exception("OpenAI API key is not available.")
            
            # Use direct HTTP requests to bypass client initialization issues
            import requests
            
            headers = {
                'Authorization': f'Bearer {api_key}',
            }
            
            # Determine correct MIME type based on file extension
            file_ext = os.path.splitext(chunk_filepath)[1].lower()
            if file_ext == '.mp3':
                mime_type = 'audio/mpeg'
            elif file_ext == '.m4a':
                mime_type = 'audio/mp4'
            elif file_ext == '.mp4':
                mime_type = 'video/mp4'
            elif file_ext == '.wav':
                mime_type = 'audio/wav'
            elif file_ext == '.flac':
                mime_type = 'audio/flac'
            elif file_ext == '.ogg':
                mime_type = 'audio/ogg'
            else:
                mime_type = 'audio/mpeg'  # Default fallback
            
            files = {
                'file': (os.path.basename(chunk_filepath), audio_file, mime_type),
                'model': (None, 'whisper-1'),
                'response_format': (None, 'verbose_json'),
            }
            
            response = requests.post(
                'https://api.openai.com/v1/audio/transcriptions',
                headers=headers,
                files=files,
                timeout=300
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
            
            transcript = response.json()
        
        # Process and adjust timestamps
        segments = []
        if 'segments' in transcript and transcript['segments']:
            for segment in transcript['segments']:
                # Direct API response uses dictionary format
                start_time = segment['start'] + time_offset
                end_time = segment['end'] + time_offset
                text = segment['text'].strip()
                
                adjusted_segment = {
                    'start': start_time,
                    'end': end_time,
                    'text': text
                }
                segments.append(adjusted_segment)
        
        # Append results to transcript file immediately
        with open(transcript_path, "a", encoding='utf-8') as f:
            if chunk_index == 0:
                # Write header for first chunk
                f.write(f"\n--- Chunk {chunk_index + 1} ({os.path.basename(chunk_filepath)}) ---\n")
            else:
                f.write(f"\n--- Chunk {chunk_index + 1} ({os.path.basename(chunk_filepath)}) ---\n")
            
            if segments:
                for segment in segments:
                    start_time = format_time(segment['start'])
                    end_time = format_time(segment['end'])
                    text = segment['text']
                    f.write(f"{start_time} - {end_time}: {text}\n")
            else:
                # Fallback if segments are not available
                f.write(f"Text: {transcript.get('text', 'No text available')}\n")
            
            f.flush()  # Ensure data is written immediately
        
        if file_id:
            progress = 40 + ((chunk_index + 1) / total_chunks) * 50
            update_status(file_id, 'transcribing', progress, f'Completed chunk {chunk_index + 1}/{total_chunks} - {len(segments)} segments added to transcript')
        
        print(f"✅ Chunk {chunk_index + 1} completed: {len(segments)} segments transcribed and appended")
        
        return {
            'text': transcript.get('text', ''),
            'segments': segments,
            'chunk_completed': True
        }
        
    except Exception as e:
        if file_id:
            update_status(file_id, 'error', 0, f'Transcription failed for chunk {chunk_index + 1}: {str(e)}')
        
        # Append error info to transcript file
        with open(transcript_path, "a", encoding='utf-8') as f:
            f.write(f"\n--- Chunk {chunk_index + 1} - ERROR ---\n")
            f.write(f"Error: {str(e)}\n")
        
        raise Exception(f"OpenAI transcription error for chunk {chunk_index + 1}: {str(e)}")

def transcribe_audio(filepath, file_id):
    """Transcribe audio file using OpenAI API with optimized 10MB chunking for large files"""
    try:
        file_size_mb = get_file_size_mb(filepath)
        print(f"📊 File size: {file_size_mb:.2f} MB")
        
        update_status(file_id, 'processing', 12, f'Starting transcription process for {file_size_mb:.1f}MB file')
        
        # Generate transcript filename and path
        transcript_filename = f"{file_id}_transcript.txt"
        transcript_path = os.path.join(app.config['TRANSCRIPTS_FOLDER'], transcript_filename)
        
        # Initialize transcript file with header
        with open(transcript_path, "w", encoding='utf-8') as f:
            f.write("Audio Transcription\n")
            f.write("=" * 50 + "\n")
            f.write(f"File size: {file_size_mb:.2f} MB\n")
            f.write(f"Processing strategy: {'10MB chunks' if file_size_mb > 20 else 'Direct processing'}\n")
            f.write(f"Processing started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
        
        # Check if file needs to be split
        print(f"🔍 Checking if file needs chunking: {file_size_mb:.2f}MB vs 20MB threshold")
        if file_size_mb > 20:  # New threshold: 20MB
            print(f"📂 File exceeds 20MB limit ({file_size_mb:.1f}MB > 20MB), splitting into 10MB chunks...")
            chunk_files = split_audio_file_by_size(filepath, target_size_mb=10, file_id=file_id)
            print(f"🔢 Chunking result: {len(chunk_files)} chunk(s) created")
            
            # Validate that chunking actually worked
            if len(chunk_files) == 1 and chunk_files[0] == filepath:
                print(f"⚠️  Chunking failed, falling back to direct processing")
                chunking_failed = True
            else:
                # Verify chunk files actually exist
                valid_chunks = []
                for chunk_file in chunk_files:
                    if os.path.exists(chunk_file):
                        valid_chunks.append(chunk_file)
                    else:
                        print(f"❌ Chunk file not found: {chunk_file}")
                
                if len(valid_chunks) == 0:
                    print(f"⚠️  No valid chunk files found, falling back to direct processing")
                    chunk_files = [filepath]
                    chunking_failed = True
                else:
                    chunk_files = valid_chunks
                    chunking_failed = False
        else:
            print(f"📝 File is small enough ({file_size_mb:.1f}MB <= 20MB) - processing directly")
            chunk_files = [filepath]
            chunking_failed = False
            update_status(file_id, 'processing', 20, f'File size OK ({file_size_mb:.1f}MB) - Processing directly')
        
        # Process each chunk sequentially and append results
        total_segments = 0
        chunk_duration_seconds = 0  # Will be calculated dynamically
        chunks_folder = None
        
        try:
            for i, chunk_file in enumerate(chunk_files):
                # Calculate time offset based on actual chunk durations
                if AUDIO_PROCESSING_AVAILABLE and len(chunk_files) > 1 and not chunking_failed:
                    if i == 0:
                        time_offset = 0
                    else:
                        # For byte-level chunks or direct processing, handle time offset
                        if len(chunk_files) > 1 and not chunking_failed:
                            # Estimate duration per chunk (this is approximate for byte-level chunks)
                            total_duration_estimate = file_size_mb * 60 * 2  # Rough estimate: 2 min per MB
                            time_offset = (i / len(chunk_files)) * total_duration_estimate
                        else:
                            # Direct processing (no chunking) - no time offset needed
                            time_offset = 0
                
                # Transcribe chunk and append to file
                chunk_result = transcribe_and_append_chunk(
                    chunk_file, i, time_offset, transcript_path, file_id, len(chunk_files)
                )
                
                total_segments += len(chunk_result['segments'])
                
                # Clean up chunk file after processing (but keep the folder for now)
                if chunk_file != filepath and os.path.exists(chunk_file):
                    if chunks_folder is None:
                        chunks_folder = os.path.dirname(chunk_file)
                    print(f"🗑️  Cleaning up processed chunk: {os.path.basename(chunk_file)}")
                    os.remove(chunk_file)
            
            # Clean up chunks folder if it was created
            if chunks_folder and os.path.exists(chunks_folder):
                try:
                    os.rmdir(chunks_folder)
                    print(f"🗑️  Cleaned up chunks folder: {chunks_folder}")
                except OSError:
                    print(f"⚠️  Could not remove chunks folder (may not be empty): {chunks_folder}")
            
            update_status(file_id, 'finalizing', 92, f'Finalizing transcript with {total_segments} total segments...')
            
            # Append final summary to transcript file
            with open(transcript_path, "a", encoding='utf-8') as f:
                f.write("\n" + "=" * 50 + "\n")
                f.write(f"Processing completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total chunks processed: {len(chunk_files)}\n")
                f.write(f"Total segments transcribed: {total_segments}\n")
                f.write("=" * 50 + "\n")
            
            update_status(file_id, 'completed', 100, f'Transcription completed! Processed {len(chunk_files)} chunk(s) with {total_segments} total segments')
            print(f"✅ Transcription complete: {transcript_filename}")
            return transcript_filename
            
        except Exception as chunk_error:
            # Clean up any remaining chunk files on error
            for chunk_file in chunk_files:
                if chunk_file != filepath and os.path.exists(chunk_file):
                    try:
                        os.remove(chunk_file)
                    except:
                        pass
            
            # Clean up chunks folder
            if chunks_folder and os.path.exists(chunks_folder):
                try:
                    os.rmdir(chunks_folder)
                except:
                    pass
            
            raise chunk_error
        
    except Exception as e:
        update_status(file_id, 'error', 0, f'Transcription failed: {str(e)}')
        
        # Append error to transcript file if it exists
        transcript_filename = f"{file_id}_transcript.txt"
        transcript_path = os.path.join(app.config['TRANSCRIPTS_FOLDER'], transcript_filename)
        if os.path.exists(transcript_path):
            with open(transcript_path, "a", encoding='utf-8') as f:
                f.write(f"\n\nERROR: {str(e)}\n")
        
        raise Exception(f"OpenAI transcription error: {str(e)}")

def format_time(seconds):
    """Format seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

@app.route('/download/<transcript_id>')
def download_transcript(transcript_id):
    """Download transcript file"""
    try:
        transcript_filename = f"{transcript_id}_transcript.txt"
        transcript_path = os.path.join(app.config['TRANSCRIPTS_FOLDER'], transcript_filename)
        
        if os.path.exists(transcript_path):
            return send_file(
                transcript_path,
                as_attachment=True,
                download_name=f"transcript_{transcript_id}.txt",
                mimetype='text/plain'
            )
        else:
            return jsonify({'error': 'Transcript not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Health check endpoint"""
    return jsonify({'status': 'running', 'message': 'Audio Transcription Service is operational'})

@app.route('/admin')
def admin_dashboard():
    """Admin management dashboard"""
    return render_template('admin.html')

@app.route('/api/stats')
def get_stats():
    """Get application statistics"""
    try:
        # Count transcript files
        transcript_files = [f for f in os.listdir(app.config['TRANSCRIPTS_FOLDER']) if f.endswith('.txt')]
        total_transcriptions = len(transcript_files)
        
        # Calculate storage used
        storage_used = 0
        for filename in transcript_files:
            filepath = os.path.join(app.config['TRANSCRIPTS_FOLDER'], filename)
            if os.path.exists(filepath):
                storage_used += os.path.getsize(filepath)
        storage_used_mb = round(storage_used / (1024 * 1024), 2)
        
        return jsonify({
            'total': total_transcriptions,
            'successful': total_transcriptions,  # Assuming all completed files are successful
            'avgTime': 45,  # Placeholder - could be tracked in future
            'storage': storage_used_mb
        })
    except Exception as e:
        return jsonify({
            'total': 0,
            'successful': 0,
            'avgTime': 0,
            'storage': 0
        })

@app.route('/api/transcriptions')
def get_transcriptions():
    """Get all transcriptions"""
    try:
        transcriptions = []
        transcript_files = [f for f in os.listdir(app.config['TRANSCRIPTS_FOLDER']) if f.endswith('.txt')]
        
        for filename in transcript_files:
            filepath = os.path.join(app.config['TRANSCRIPTS_FOLDER'], filename)
            if os.path.exists(filepath):
                stat = os.stat(filepath)
                file_id = filename.replace('_transcript.txt', '')
                
                transcriptions.append({
                    'id': file_id,
                    'filename': filename,
                    'size': f"{round(stat.st_size / 1024, 2)} KB",
                    'status': 'completed',
                    'created': datetime.datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sort by creation time (newest first)
        transcriptions.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify(transcriptions)
    except Exception as e:
        return jsonify([])

@app.route('/api/recent-transcriptions')
def get_recent_transcriptions():
    """Get recent transcriptions (last 5)"""
    try:
        all_transcriptions = get_transcriptions().get_json()
        recent = all_transcriptions[:5] if all_transcriptions else []
        
        # Add duration placeholder
        for item in recent:
            item['duration'] = '2:30'  # Placeholder - could be extracted from audio metadata
        
        return jsonify(recent)
    except Exception as e:
        return jsonify([])

@app.route('/api/transcriptions/<transcript_id>', methods=['DELETE'])
def delete_transcription(transcript_id):
    """Delete a transcription"""
    try:
        transcript_filename = f"{transcript_id}_transcript.txt"
        transcript_path = os.path.join(app.config['TRANSCRIPTS_FOLDER'], transcript_filename)
        
        if os.path.exists(transcript_path):
            os.remove(transcript_path)
            return jsonify({'success': True, 'message': 'Transcription deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Transcription not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error deleting transcription: {str(e)}'}), 500

@app.route('/api/test-connection')
def test_api_connection():
    """Test OpenAI API connection"""
    if client is None:
        return jsonify({'success': False, 'message': 'OpenAI client not initialized. Check API key.'}), 500
    
    try:
        # Test with a simple API call
        models = client.models.list()
        return jsonify({'success': True, 'message': 'OpenAI API connection successful'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'API connection failed: {str(e)}'}), 500

@app.route('/view/<transcript_id>')
def view_transcript(transcript_id):
    """View transcript content in browser"""
    try:
        transcript_filename = f"{transcript_id}_transcript.txt"
        transcript_path = os.path.join(app.config['TRANSCRIPTS_FOLDER'], transcript_filename)
        
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Return as HTML with basic formatting
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Transcript - {transcript_id}</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body>
                <div class="container mt-4">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h2>Transcript View</h2>
                        <a href="/download/{transcript_id}" class="btn btn-primary">
                            <i class="fas fa-download me-1"></i>Download
                        </a>
                    </div>
                    <div class="card">
                        <div class="card-body">
                            <pre class="mb-0" style="white-space: pre-wrap; font-family: 'Courier New', monospace;">{content}</pre>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            return html_content
        else:
            return jsonify({'error': 'Transcript not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Error viewing transcript: {str(e)}'}), 500

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    if get_openai_client() is None:
        return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.'}), 500

    data = request.get_json()
    text = data.get('text')
    voice = data.get('voice', 'alloy')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        api_key = get_openai_client()
        client = OpenAI(api_key=api_key)

        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio_file:
            response.stream_to_file(tmp_audio_file.name)
            tmp_audio_file.seek(0)
            return send_file(tmp_audio_file.name, as_attachment=True, download_name='generated_audio.mp3', mimetype='audio/mpeg')

    except Exception as e:
        return jsonify({'error': f'Failed to generate audio: {str(e)}'}), 500

@app.route('/generate-image', methods=['POST'])
def generate_image():
    if get_openai_client() is None:
        return jsonify({'error': 'OpenAI API key not configured.'}), 500

    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        api_key = get_openai_client()
        client = OpenAI(api_key=api_key)

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        
        image_url = response.data[0].url
        return jsonify({'image_url': image_url})

    except Exception as e:
        return jsonify({'error': f'Failed to generate image: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if get_openai_client() is None:
        return jsonify({'error': 'OpenAI API key not configured.'}), 500

    data = request.get_json()
    history = data.get('history')

    if not history:
        return jsonify({'error': 'No chat history provided'}), 400

    try:
        api_key = get_openai_client()
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=history
        )

        reply = response.choices[0].message.content
        return jsonify({'reply': reply})

    except Exception as e:
        return jsonify({'error': f'Failed to get chat response: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
