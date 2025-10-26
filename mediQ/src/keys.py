from dotenv import load_dotenv
import os
load_dotenv()
mykey = {
    "mediQ": "sk-1234567890abcdef1234567890abcdef12345678",
    # DeepSeek key placeholder; update this value with your real key before running.
    "deepseek": os.environ.get('DEEPSEEK_API_KEY'),
}
