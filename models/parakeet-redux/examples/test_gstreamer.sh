#!/bin/sh
# End-to-end tests of gst_stdin.py and gst_appsink.py, file and live G.711 RTP.
# Run inside an image with GStreamer, PyGObject and sherpa-onnx (Debian bookworm:
# gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good python3-gi
# gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0, then a venv with
# --system-site-packages and `pip install sherpa-onnx numpy`), with this directory
# at /ex, a 16 kHz mono test WAV at /t/clip90.wav, the models at /models and an
# output directory at /out. The committed log was made from the first 90 s of
# AppTek en_US_General_Agriculture_1586590_channel1.wav.
export MODEL_DIR=/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3 VAD_MODEL=/models/silero_vad_v5.onnx MODELS_DIR=/models
cd /ex
CAPS='caps=application/x-rtp,media=audio,clock-rate=8000,encoding-name=PCMU,payload=0'
SEND='filesrc location=/t/clip90.wav ! wavparse ! audioconvert ! audioresample ! audio/x-raw,rate=8000,channels=1 ! mulawenc ! rtppcmupay ! udpsink host=127.0.0.1 port=5004'
TAIL='! audioconvert ! audioresample ! audio/x-raw,format=S16LE,rate=16000,channels=1'
RTP="udpsrc port=5004 $CAPS ! rtpjitterbuffer latency=60 ! rtppcmudepay ! mulawdec $TAIL"

echo "== 1 gst_stdin.py, file"
gst-launch-1.0 -q filesrc location=/t/clip90.wav ! decodebin $TAIL ! fdsink fd=1 | /venv/bin/python gst_stdin.py > /out/1.txt
wc -l < /out/1.txt

echo "== 2 gst_stdin.py, live G.711 RTP (sent in real time, 90 s)"
( timeout -s INT 97 gst-launch-1.0 -q -e $RTP ! fdsink fd=1 | /venv/bin/python gst_stdin.py > /out/2.txt ) &
sleep 3; t0=$(date +%s); gst-launch-1.0 -q $SEND; echo "   sender ran $(( $(date +%s) - t0 )) s"; wait
wc -l < /out/2.txt

echo "== 3 gst_appsink.py, file"
/venv/bin/python gst_appsink.py "filesrc location=/t/clip90.wav ! decodebin $TAIL ! appsink name=sink" > /out/3.txt
wc -l < /out/3.txt

echo "== 4 gst_appsink.py, live G.711 RTP, stopped with SIGINT"
( timeout -s INT 97 /venv/bin/python gst_appsink.py "$RTP ! appsink name=sink" > /out/4.txt ) &
sleep 5; gst-launch-1.0 -q $SEND; wait
wc -l < /out/4.txt
echo "== 5 nemotron_stream.py (streaming model), file"
gst-launch-1.0 -q filesrc location=/t/clip90.wav ! decodebin $TAIL ! fdsink fd=1 | /venv/bin/python nemotron_stream.py > /out/5.txt
grep -c . /out/5.txt

echo "== 6 nemotron_stream.py, live G.711 RTP (sent in real time, 90 s)"
( timeout -s INT 97 gst-launch-1.0 -q -e $RTP ! fdsink fd=1 | /venv/bin/python nemotron_stream.py > /out/6.txt ) &
sleep 3; gst-launch-1.0 -q $SEND; wait
grep -c . /out/6.txt

echo "== identical: file stdin vs appsink: $(cmp -s /out/1.txt /out/3.txt && echo yes || echo no); RTP stdin vs appsink: $(cmp -s /out/2.txt /out/4.txt && echo yes || echo no)"
