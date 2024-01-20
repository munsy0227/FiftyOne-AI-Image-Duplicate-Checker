import os
from PIL import Image
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from sklearn.metrics.pairwise import cosine_similarity

# 사용자로부터 추천값과 이미지 경로를 입력받음
# 사용자로부터 추천값을 입력받음, 입력이 없을 경우 기본값은 0.985
similarity_threshold = float(input("추천값은 0.985 (엔터 키 입력 시 추천값): ") or "0.985")
images_folder = os.path.abspath(input("이미지 경로 입력: "))
root_dir = os.getcwd()

# 현재 디렉토리를 저장하고 작업 디렉토리를 이동함
os.chdir(root_dir)

# 모델 및 변수 초기화
model_name = "clip-vit-base32-torch"
supported_types = (".png", ".jpg", ".jpeg", ".webp")
img_count = len(os.listdir(images_folder))
batch_size = min(250, img_count)

# 비 이미지 파일 및 이미지 없는 경우 오류 출력
non_images = [f for f in os.listdir(images_folder) if not f.lower().endswith(supported_types)]
if non_images:
    print(f"💥 Error: Found non-image file {non_images[0]} - This program doesn't allow it. Sorry! Use the Extras at the bottom to clean the folder.")
elif img_count == 0:
    print(f"💥 Error: No images found in {images_folder}")
else:
    print("\n💿 Analyzing dataset...\n")
    # FiftyOne 데이터셋 초기화
    dataset = fo.Dataset()
    for image_file in os.listdir(images_folder):
        if image_file.lower().endswith(supported_types):
            image_path = os.path.join(images_folder, image_file)
            sample = fo.Sample(filepath=image_path)
            dataset.add_sample(sample)

    # FiftyOne Zoo 모델 로드 및 임베딩 계산
    model = foz.load_zoo_model(model_name)
    embeddings = dataset.compute_embeddings(model, batch_size=batch_size)

    # 이미지 임베딩을 배치로 나누고 코사인 유사도 행렬 계산
    batch_embeddings = np.array_split(embeddings, batch_size)
    similarity_matrices = []
    max_size_x = max(array.shape[0] for array in batch_embeddings)
    max_size_y = max(array.shape[1] for array in batch_embeddings)

    for i, batch_embedding in enumerate(batch_embeddings):
        similarity = cosine_similarity(batch_embedding)
        # 0으로 패딩하여 np.concatenate에 사용
        padded_array = np.zeros((max_size_x, max_size_y))
        padded_array[0:similarity.shape[0], 0:similarity.shape[1]] = similarity
        similarity_matrices.append(padded_array)

    similarity_matrix = np.concatenate(similarity_matrices, axis=0)
    similarity_matrix = similarity_matrix[0:embeddings.shape[0], 0:embeddings.shape[0]]

    # 코사인 유사도 행렬 계산 및 유사도 임계값에 따라 이미지 매칭
    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix -= np.identity(len(similarity_matrix))

    dataset.match(F("max_similarity") > similarity_threshold)
    dataset.tags = ["delete", "has_duplicates"]

    id_map = [s.id for s in dataset.select_fields(["id"])]
    samples_to_remove = set()
    samples_to_keep = set()

    for idx, sample in enumerate(dataset):
        if sample.id not in samples_to_remove:
            # 두 개의 중복된 인스턴스 중 첫 번째 인스턴스 유지
            samples_to_keep.add(sample.id)
            
            dup_idxs = np.where(similarity_matrix[idx] > similarity_threshold)[0]

            for dup in dup_idxs:
                # 첫 번째 인스턴스를 유지했으므로 나머지 중복된 항목을 모두 제거
                    samples_to_remove.add(id_map[dup])

            if len(dup_idxs) > 0:
                sample.tags.append("has_duplicates")
                sample.save()
        else:
            sample.tags.append("delete")
            sample.save()

    # FiftyOne 앱 실행 및 사용자 지침 출력
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

    # 삭제할 이미지 제거 및 결과 저장
    kys = [s for s in dataset if "delete" in s.tags]
    dataset.remove_samples(kys)
    os.mkdir(path=os.path.join(images_folder, "output"))
    OUTPUT = os.path.abspath(os.path.join(images_folder, "output"))
    dataset.export(export_dir=OUTPUT, dataset_type=fo.types.ImageDirectory)

    # 세션 및 앱 닫기
    session.refresh()
    fo.close_app()

    print(f"\n✅ Removed {len(kys)} images from dataset. You now have {len(os.listdir(OUTPUT))} images.")

