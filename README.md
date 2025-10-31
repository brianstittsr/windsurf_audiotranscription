# Audio Transcription Flask App

A Flask web application that transcribes audio files (MP3, WAV, M4A, FLAC, OGG) to text using OpenAI's Whisper API with timestamp information.

## Features

- 🎵 Upload audio files via drag & drop or file selection
- 🤖 Automatic transcription using OpenAI Whisper API
- ⏱️ Timestamp information for each segment
- 📥 Download transcripts as text files
- 🎨 Modern, responsive web interface
- 📱 Mobile-friendly design

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

You need to set your OpenAI API key as an environment variable:

**Windows:**
```cmd
set OPENAI_API_KEY=your_api_key_here
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY=your_api_key_here
```

Or create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Open your web browser and go to `http://localhost:5000`
2. Upload an audio file by:
   - Clicking the upload area and selecting a file
   - Dragging and dropping a file onto the upload area
3. Click "Start Transcription" to begin processing
4. Wait for the transcription to complete
5. Download the transcript text file

## Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- OGG

## File Size Limits

- Maximum file size: 100MB
- Files are temporarily stored during processing and automatically deleted afterward

## API Endpoints

- `GET /` - Main application interface
- `POST /upload` - Upload and transcribe audio file
- `GET /download/<transcript_id>` - Download transcript file
- `GET /status` - Health check endpoint

## Project Structure

```
AudioTranscription/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Temporary audio file storage
├── transcripts/          # Generated transcript files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Security Notes

- Change the `SECRET_KEY` in `app.py` for production use
- Ensure your OpenAI API key is kept secure and not committed to version control
- Consider implementing user authentication for production deployments

## Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Make sure you've set the `OPENAI_API_KEY` environment variable

2. **"File too large"**
   - The maximum file size is 100MB. Try compressing your audio file

3. **"Unsupported file format"**
   - Only MP3, WAV, M4A, FLAC, and OGG files are supported

4. **Transcription fails**
   - Check your OpenAI API key and account credits
   - Ensure the audio file is not corrupted

## License

This project is open source and available under the MIT License.
