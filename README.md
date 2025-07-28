# Phone Finder: Smart Phone Search & Recommendation Engine

This project is a web application that helps users find the perfect smartphone based on their specific needs and budget. It features a clean, interactive user interface and a backend powered by Flask and a machine learning model. Users can search for phones by brand, set a maximum budget, and filter by minimum RAM and ROM requirements.

## 🚀 Features

-   **Dynamic Search:** Search for phones by brand name (e.g., "Samsung", "Apple", "OnePlus").
-   **Advanced Filtering:**
    -   Set a maximum price to find phones within your budget.
    -   Specify minimum RAM (in GB) for performance needs.
    -   Specify minimum ROM (in GB) for storage needs.
-   **Interactive UI:** A modern and visually appealing frontend built with HTML and Tailwind CSS, featuring animations and a responsive design.
-   **Wishlist:** Add your favorite phones to a personal wishlist for later viewing.
-   **Data-Driven:** Powered by a dataset of smartphones, which is preprocessed and cleaned using a data science pipeline.

## 🛠️ Tech Stack

-   **Backend:** Python, Flask
-   **Frontend:** HTML, Tailwind CSS, JavaScript
-   **Data Science & ML:** Pandas, NumPy, Scikit-learn, NLTK

## 📂 Project Structure

```
.
├── main.py                   # Flask server application
├── data_preprocessing.ipynb  # Jupyter Notebook for data cleaning and model logic
├── templates/
│   └── index.html            # Frontend HTML file
└── smart_phones1.csv         # The raw dataset
```

-   **`main.py`**: The core of the web application. It handles HTTP requests, processes user input, filters the phone data, and renders the frontend.
-   **`data_preprocessing.ipynb`**: Contains all the data cleaning, feature engineering, and vectorization steps. This notebook is where the logic for understanding the phone data is developed.
-   **`templates/index.html`**: A single-page application that provides the user interface for searching, filtering, and viewing phone results.
-   **`smart_phones1.csv`**: The dataset containing information about various smartphones, including their specifications and pricing.

## ⚙️ How It Works

The project follows a standard machine learning web application architecture.

### 1. Data Preprocessing (`data_preprocessing.ipynb`)

The raw `smart_phones1.csv` data is messy. The Jupyter Notebook performs several crucial steps to make it usable:

-   **Data Cleaning:** Extracts and standardizes inconsistent data from columns like `RAM`, `ROM`, `Display_Size`, and `Current_Price`. For example, it removes currency symbols and converts storage units (MB to GB).
-   **Feature Engineering:** New, more useful features like `Brand` and `Colour` are extracted from the `Phone_Name` column.
-   **Handling Missing Values:** The `KNNImputer` from Scikit-learn is used to intelligently fill in missing numerical data (like `Battery` or `Price`) based on the values of similar phones.
-   **Outlier Removal:** Z-score is used to identify and cap extreme outlier values in numerical columns to prevent them from skewing the data.

### 2. Vectorization & Similarity Model (`data_preprocessing.ipynb`)

To build a foundation for a recommendation system, the notebook also includes:

-   **Text Combination:** All key specifications of a phone are combined into a single text string called `details`.
-   **Stemming:** NLTK's `PorterStemmer` is used to reduce words to their root form (e.g., "processors" becomes "processor").
-   **Vectorization:** `CountVectorizer` converts the `details` string for each phone into a numerical vector.
-   **Cosine Similarity:** `cosine_similarity` is calculated between all phone vectors. This creates a similarity matrix that can identify which phones are most alike based on their specs. *Note: The current web app uses direct filtering, but this similarity matrix is built for future recommendation features.*

### 3. Flask Web Application (`main.py`)

-   **Data Loading:** On startup, the server loads and preprocesses the `smart_phones1.csv` file.
-   **Routing:** It defines routes for the main page (`/`) and for handling search queries (`/search`).
-   **Search & Filter Logic:** When a user submits the search form, the backend receives the criteria (brand, budget, etc.), filters the preprocessed Pandas DataFrame accordingly, and passes the results back to the frontend.
-   **Wishlist API:** Endpoints are provided to add, remove, and retrieve items from a global wishlist.

## 📦 Setup and Installation

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sidbhaskar/Phone_Finder_Project.git
    cd Phone_Finder_Project
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    # Create a virtual environment
    python -m venv venv
    # Activate it (Windows)
    .\venv\Scripts\activate
    # Or (macOS/Linux)
    source venv/bin/activate

    # Install the required libraries
    pip install Flask pandas numpy scikit-learn nltk
    ```

3.  **Run the Flask application:**
    ```bash
    python main.py
    ```

4.  **Open in browser:**
    Navigate to `http://127.0.0.1:5000` in your web browser to use the application.

## 💡 Future Improvements

-   [ ] **Implement a Recommendation Engine:** Use the pre-calculated `cosine_similarity` matrix to add a "You might also like..." section on a product detail page.
-   [ ] **User Accounts:** Add user authentication so each user can have their own private wishlist.
-   [ ] **Database Integration:** Store the phone data and user information in a database like SQLite or PostgreSQL for better scalability and persistence.
-   [ ] **Deployment:** Deploy the application to a cloud platform like Heroku or AWS so it's publicly accessible.
