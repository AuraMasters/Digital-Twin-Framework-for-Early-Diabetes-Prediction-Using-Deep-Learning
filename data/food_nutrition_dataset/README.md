# Food Nutrition Image Dataset & Test Benchmark

## 1. Overview
This dataset contains high-resolution culinary meal images across diverse meal categories (Breakfast, Lunch, Dinner, Snack) paired with granular USDA-calibrated continuous macronutrient ground-truth targets and categorical labels.

---

## 2. Directory Structure

```
data/food_nutrition_dataset/
├── metadata.csv                     # Comprehensive nutritional ground truth labels
├── README.md                        # Dataset documentation & target schema
├── images/                          # High-resolution food images
│   ├── dish_001_salad.jpg
│   ├── dish_002_oatmeal.jpg
│   ├── dish_003_avocado_toast.jpg
│   ├── dish_004_yogurt_parfait.jpg
│   ├── dish_005_salmon_plate.jpg
│   ├── dish_006_burger.jpg
│   ├── dish_007_pasta.jpg
│   ├── dish_008_steak.jpg
│   ├── dish_009_veggie_bowl.jpg
│   ├── dish_010_smoothie_bowl.jpg
│   ├── dish_011_pizza.jpg
│   ├── dish_012_sandwich.jpg
│   ├── dish_013_ramen_noodles.jpg
│   ├── dish_014_protein_pancake.jpg
│   └── dish_015_green_salad.jpg
└── test_samples/                    # Dedicated test evaluation samples
    ├── test_sample_01_oatmeal.jpg
    ├── test_sample_02_salmon.jpg
    ├── test_sample_03_pasta.jpg
    ├── test_sample_04_smoothie.jpg
    └── test_sample_05_salad.jpg
```

---

## 3. Ground Truth Nutritional Variables (`metadata.csv`)

| Variable | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `dish_id` | String | - | Unique identifier for dish |
| `food_name` | String | - | Descriptive dish name |
| `meal_type` | Categorical | - | Meal category (`Breakfast`, `Lunch`, `Dinner`, `Snack`) |
| `food_tag` | Categorical | - | Food taxonomy item (`Salad`, `Oatmeal`, `Toast`, `Salmon`, `Pasta`, etc.) |
| `calories_kcal` | Continuous | $kcal$ | Total caloric energy |
| `carbohydrates_g`| Continuous | $g$ | Total carbohydrates |
| `protein_g` | Continuous | $g$ | Total dietary protein |
| `fat_g` | Continuous | $g$ | Total dietary lipid fat |
| `fiber_g` | Continuous | $g$ | Dietary fiber content |
| `sugar_g` | Continuous | $g$ | Simple sugars |
| `sodium_mg` | Continuous | $mg$ | Sodium mineral content |
| `portion_weight_g`| Continuous| $g$ | Total edible portion mass |
| `image_path` | String | - | Relative path to image file |
