from dotenv import load_dotenv
import kagglehub

load_dotenv()
kagglehub.login()

path = kagglehub.dataset_download("fraxle/images-from-fromsoftware-soulslikes")

print("Path to dataset files:", path)