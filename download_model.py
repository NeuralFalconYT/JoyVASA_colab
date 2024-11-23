from huggingface_hub import hf_hub_download, snapshot_download
import os
import shutil
import platform
def download_models():
  base_path="."
  model_folder=f"{base_path}/pretrained_weights"
  os.makedirs(model_folder,exist_ok=True)
  
  os.makedirs(f"{model_folder}/JoyVASA/motion_generator", exist_ok=True)
  os.makedirs(f"{model_folder}/JoyVASA/motion_template", exist_ok=True)
  download_folder = f"{model_folder}/JoyVASA"
  hf_hub_download(repo_id="jdh-algo/JoyVASA", filename="motion_generator/motion_generator_hubert_chinese.pt",local_dir=download_folder)
  hf_hub_download(repo_id="jdh-algo/JoyVASA", filename="motion_template/motion_template.pkl",local_dir=download_folder)


  model_repo = "facebook/wav2vec2-base-960h" 
  folder_name="wav2vec2-base-960h"
  download_folder = f"{model_folder}/{folder_name}"
  os.makedirs(download_folder,exist_ok=True)
  snapshot_path = snapshot_download(repo_id=model_repo, local_dir=download_folder)
  print(f"Snapshot downloaded to: {snapshot_path}")

  model_repo = "TencentGameMate/chinese-hubert-base"  
  
  if platform.system() == "Windows":
    folder_name="chinese-hubert-base"
  else:
    folder_name="TencentGameMate:chinese-hubert-base"
  download_folder = f"{model_folder}/{folder_name}"
  os.makedirs(download_folder,exist_ok=True)
  snapshot_path = snapshot_download(repo_id=model_repo, local_dir=download_folder)
  print(f"Snapshot downloaded to: {snapshot_path}")

  model_repo = "KwaiVGI/LivePortrait"  
  os.makedirs(f"{model_folder}/docs", exist_ok=True)
  os.makedirs(f"{model_folder}/insightface/models/buffalo_l", exist_ok=True)
  os.makedirs(f"{model_folder}/liveportrait/base_models", exist_ok=True)
  os.makedirs(f"{model_folder}/liveportrait/retargeting_models", exist_ok=True)
  os.makedirs(f"{model_folder}/liveportrait_animals/base_models", exist_ok=True)
  os.makedirs(f"{model_folder}/liveportrait_animals/retargeting_models", exist_ok=True)
  snapshot_path = snapshot_download(repo_id=model_repo, local_dir=model_folder)
  for i in ["/README.md","/docs/inference.gif","/docs/showcase2.gif","/.gitignore","/.gitattributes","/.gitkeep"]:
    delete_path=f"{model_folder}{i}"
    if os.path.exists(delete_path):
      os.remove(delete_path)
    if os.path.exists(f"{model_folder}/docs"):
      shutil.rmtree(f"{model_folder}/docs")
download_models()
#python download_model.py
