# Vercel Deployment Guide

## Prerequisites
- Vercel account
- GitHub repository connected to Vercel
- OpenAI API key

## Deployment Steps

### 1. Connect Repository to Vercel
1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "Add New" → "Project"
3. Import your GitHub repository: `brianstittsr/windsurf_audiotranscription`

### 2. Configure Environment Variables
In Vercel project settings, add the following environment variable:
- `OPENAI_API_KEY`: Your OpenAI API key

### 3. Deploy
Vercel will automatically deploy when you push to your repository.

## Important Notes for Flask on Vercel

### Limitations
- **File uploads**: Vercel has a 4.5MB body size limit for serverless functions
- **Execution time**: 10 seconds for Hobby plan, 60 seconds for Pro
- **File system**: Read-only except for `/tmp` directory

### Recommended Solutions
For production deployment with large file uploads:
1. Use a dedicated hosting service (e.g., Railway, Render, Heroku)
2. Or implement direct-to-S3 uploads with presigned URLs
3. Or use Vercel Edge Functions with streaming

### Current Configuration
- `vercel.json`: Routes all requests to Flask app
- `app.py`: Exports `application` variable for WSGI compatibility
- All routes configured for serverless deployment

## Local Development
```bash
python app.py
```
Access at: http://localhost:5001

## Vercel-Specific Issues

### Issue: Flask app not loading
**Solution**: Ensure `vercel.json` is properly configured and `application = app` is set in `app.py`

### Issue: File uploads failing
**Solution**: This is a Vercel limitation. Consider:
- Using external storage (S3, Cloudinary)
- Deploying to a different platform for file-heavy operations
- Implementing chunked uploads

### Issue: Environment variables not loading
**Solution**: Set environment variables in Vercel dashboard, not in `.env` file (which is gitignored)

## Alternative Deployment Platforms

For this application's requirements (large file uploads, long processing times), consider:
- **Railway**: https://railway.app
- **Render**: https://render.com
- **Fly.io**: https://fly.io
- **DigitalOcean App Platform**: https://www.digitalocean.com/products/app-platform

These platforms better support:
- Larger file uploads
- Longer execution times
- Persistent storage
- Background jobs
