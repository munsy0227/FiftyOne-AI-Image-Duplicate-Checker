import os

similarity_threshold = float(input("추천값은 0.985: "))
images_folder1 = str(input("이미지 경로 입력: "))
images_folder = os.path.abspath(images_folder1)
root_dir = os.getcwd()

os.chdir(root_dir)
model_name = "clip-vit-base32-torch"
supported_types = (".png", ".jpg", ".jpeg", ".webp")
img_count = len(os.listdir(images_folder))
batch_size = min(250, img_count)

from PIL import Image
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from sklearn.metrics.pairwise import cosine_similarity

non_images = [f for f in os.listdir(images_folder) if not f.lower().endswith(supported_types)]
if non_images:
  print(f"💥 Error: Found non-image file {non_images[0]} - This program doesn't allow it. Sorry! Use the Extras at the bottom to clean the folder.")
elif img_count == 0:
  print(f"💥 Error: No images found in {images_folder}")
else:
  print("\n💿 Analyzing dataset...\n")
  dataset = fo.Dataset()
  for image_file in os.listdir(images_folder):
        if image_file.lower().endswith(supported_types):
            image_path = os.path.join(images_folder, image_file)
            sample = fo.Sample(filepath=image_path)
            dataset.add_sample(sample)
  model = foz.load_zoo_model(model_name)
  embeddings = dataset.compute_embeddings(model, batch_size=batch_size)

  batch_embeddings = np.array_split(embeddings, batch_size)
  similarity_matrices = []
  max_size_x = max(array.shape[0] for array in batch_embeddings)
  max_size_y = max(array.shape[1] for array in batch_embeddings)

  for i, batch_embedding in enumerate(batch_embeddings):
    similarity = cosine_similarity(batch_embedding)
    #Pad 0 for np.concatenate
    padded_array = np.zeros((max_size_x, max_size_y))
    padded_array[0:similarity.shape[0], 0:similarity.shape[1]] = similarity
    similarity_matrices.append(padded_array)

  similarity_matrix = np.concatenate(similarity_matrices, axis=0)
  similarity_matrix = similarity_matrix[0:embeddings.shape[0], 0:embeddings.shape[0]]

  similarity_matrix = cosine_similarity(embeddings)
  similarity_matrix -= np.identity(len(similarity_matrix))

  dataset.match(F("max_similarity") > similarity_threshold)
  dataset.tags = ["delete", "has_duplicates"]

  id_map = [s.id for s in dataset.select_fields(["id"])]
  samples_to_remove = set()
  samples_to_keep = set()

  for idx, sample in enumerate(dataset):
    if sample.id not in samples_to_remove:
      # Keep the first instance of two duplicates
      samples_to_keep.add(sample.id)
      
      dup_idxs = np.where(similarity_matrix[idx] > similarity_threshold)[0]
      for dup in dup_idxs:
          # We kept the first instance so remove all other duplicates
          samples_to_remove.add(id_map[dup])

      if len(dup_idxs) > 0:
          sample.tags.append("has_duplicates")
          sample.save()
    else:
      sample.tags.append("delete")
      sample.save()

#  clear_output()

  sidebar_groups = fo.DatasetAppConfig.default_sidebar_groups(dataset)
  for group in sidebar_groups[1:]:
    group.expanded = False
  dataset.app_config.sidebar_groups = sidebar_groups
  dataset.save()
  session = fo.launch_app(dataset)

  print("❗ Wait a minute for the session to load. If it doesn't, read above.")
  print("❗ When it's ready, you'll see a grid of your images.")
  print("❗ On the left side enable \"sample tags\" to visualize the images marked for deletion.")
  print("❗ You can mark your own images with the \"delete\" label by selecting them and pressing the tag icon at the top.")
  input("⭕ When you're done, enter something here to save your changes: ")

  print("💾 Saving...")

  kys = [s for s in dataset if "delete" in s.tags]
  dataset.remove_samples(kys)
  previous_folder = images_folder[:images_folder.rfind("/")]
  os.mkdir(path = images_folder1 + "/output")
  OUTPUT = os.path.abspath(images_folder1) + "/output"
  dataset.export(export_dir=os.path.join(images_folder, OUTPUT), dataset_type=fo.types.ImageDirectory)
  
  temp_suffix = "_temp"
 # !mv {images_folder} {images_folder}{temp_suffix}
 #!mv {images_folder}{temp_suffix}/{project_subfolder} {images_folder}
 #!rm -r {images_folder}{temp_suffix}

  session.refresh()
  fo.close_app()
  #clear_output()

  print(f"\n✅ Removed {len(kys)} images from dataset. You now have {len(os.listdir(OUTPUT))} images.")

