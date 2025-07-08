# Create project structure
mkdir halos-messenger
cd halos-messenger
python -m venv venv
source venv/bin/activate

# Core dependencies
pip install matrix-nio aiosqlite eth-account syft stripe cryptography