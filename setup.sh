apt-get update -y;
apt-get upgrade -y;
apt-get install python3.10-venv -y
python3 -m venv env_minihf
source env_minihf/bin/activate
pip3 install -r requirements.txt 
flask --app minihf_infer run
