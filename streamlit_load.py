from pyngrok import ngrok
import subprocess
import time
import os

 #Terminate any open ngrok tunnels from previous runs
ngrok.kill()

# IMPORTANT: For stable long-term tunnels, you might need to add your ngrok authtoken.
# You can get one from https://dashboard.ngrok.com/auth/your-authtoken
ngrok.set_auth_token("378hd2i3PcBJghX044GOzeK2bv1_6H3HQFqUjxamS6qNJow8p") # Uncomment and replace if you have one

# Start ngrok tunnel for port 8501 (where Streamlit will run)
public_url = ngrok.connect(addr="8501", proto="http")
print(f"Streamlit App URL: {public_url}")

# Define a log file for Streamlit's output
streamlit_log_file = "streamlit_output.log"

# Start Streamlit in the background, redirecting its output to the log file.
# Using 'nohup' and 'preexec_fn=os.setpgrp' helps ensure the process continues
# even if the Colab cell loses focus or completes its execution.
# '--server.headless true' prevents Streamlit from trying to open a browser window on the server.
# '--browser.gatherUsageStats false' disables sending usage statistics.
with open(streamlit_log_file, "w") as f:
    subprocess.Popen(['streamlit', 'run', 'webview.py', '--server.port', '8501', '--server.headless', 'true',
                      '--browser.gatherUsageStats', 'false'],
                     stdout=f, stderr=f, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

print(f"Streamlit logs are being written to: {streamlit_log_file}")
print("Please wait a few seconds for the Streamlit app to start completely.")
time.sleep(10) # Give Streamlit some time to initialize

print("\nIf the Streamlit app doesn't load or shows an error, you can check the logs for details:")
print(f"!cat {streamlit_log_file}")