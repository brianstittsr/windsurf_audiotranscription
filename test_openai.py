#!/usr/bin/env python3
"""
Test script to isolate OpenAI client initialization issues
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing OpenAI client initialization...")
print(f"API Key present: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

try:
    # Test 1: Import OpenAI
    print("\n1. Testing import...")
    import openai
    print(f"   OpenAI version: {openai.__version__}")
    
    # Test 2: Try basic client initialization
    print("\n2. Testing basic client initialization...")
    from openai import OpenAI
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        # Try minimal initialization
        client = OpenAI(api_key=api_key)
        print("   ✅ Basic initialization successful")
        
        # Test a simple API call (list models)
        print("\n3. Testing API connectivity...")
        models = client.models.list()
        print(f"   ✅ API call successful - found {len(models.data)} models")
        
    else:
        print("   ❌ No API key found")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    print(f"   Error type: {type(e).__name__}")
    
    # Try alternative initialization methods
    print("\n4. Trying alternative methods...")
    try:
        # Method 1: Environment variable only
        os.environ['OPENAI_API_KEY'] = api_key
        client = OpenAI()
        print("   ✅ Environment variable method successful")
    except Exception as e2:
        print(f"   ❌ Environment method failed: {e2}")
        
        try:
            # Method 2: Legacy approach
            import openai as openai_legacy
            openai_legacy.api_key = api_key
            print("   ✅ Legacy method setup successful")
        except Exception as e3:
            print(f"   ❌ Legacy method failed: {e3}")

print("\nTest complete.")
