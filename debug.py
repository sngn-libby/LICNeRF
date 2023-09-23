from src.compressproj.models import DE_Minnen2018
from src.compressproj.zoo import image_models


if __name__ == "__main__":
	model = image_models['mbt-de'](quality=4)
	model = model.to("cuda")
	print("completed")

