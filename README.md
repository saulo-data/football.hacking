---

# 🎯 Football Hacking — Football Analysis App

'**Football Hacking** is a modern football (soccer) analytics web application built with **Streamlit** and **Python**, designed to turn raw data into meaningful tactical insights and performance analysis. The goal of this project is to empower analysts, coaches, and football enthusiasts with data-driven tools to better understand matches, league trends, tactical performance, and more — all in a user-friendly interface.

This app includes dashboards that leverage match data from multiple sources (e.g., FotMob, WhoScored) stored in MongoDB, and provides interactive navigation between analysis pages such as expected goals (xG) predictions, league overviews, match analysis, relative performance metrics, pass metrics (xT), player stats, and more.'

---

## 🚀 Features

✔️ User login and personalized experience
✔️ Interactive navigation with multiple analytical modules
✔️ Match-level and league-level visualization
✔️ Advanced statistical views (e.g., Poisson model predictions)
✔️ Pass metrics and performance dashboards
✔️ Embedded links to resources and learning content

---

## 🧠 Technical Overview

The main logic of the app lives in **football_main_app.py**, which:

* Connects to a **MongoDB database** using credentials stored securely via `st.secrets`
* Provides Google login and session management
* Loads multiple pages of analytics using Streamlit’s navigation module
* Tracks user metrics in the database
* Renders a sidebar with branding, descriptions, and useful links

Dependencies for the project are listed in **requirements.txt** and include mainstream analytics and visualization libraries such as **pandas**, **matplotlib**, **mplsoccer**, **streamlit**, **altair**, **seaborn**, and **networkx** among others. ([GitHub][1])

---

## 🛠️ Installation

To get the project running locally:

### 1. Clone the repository

```bash
git clone https://github.com/saulo-data/football.hacking.git
cd football.hacking
```

### 2. Create a Python virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧩 Configuration

### Database Setup

This app uses **MongoDB** to store and read data. You will need:

1. A running MongoDB instance (Atlas, local, or other)
2. Two connection URLs:

   * `url_board` for the app’s main database
   * `url_con` for analytics collections like **fotmob_stats**, **whoscored_calendar**, **whoscored_matches**

### Streamlit Secrets

Create a file called **.streamlit/secrets.toml** with:

```toml
[url_board] = "YOUR_MONGODB_BOARD_CONNECTION_STRING"
[url_con] = "YOUR_MONGODB_ANALYTICS_CONNECTION_STRING"
```

Make sure your keys are stored securely and not committed to version control.

---

## 🚀 Running the App

Once dependencies and secrets are set:

```bash
streamlit run football_main_app.py
```

The app will start in your browser, and you'll be prompted to log in with Google to access the full functionality.

---

## 🗂️ App Structure

* `football_main_app.py` – App entry point
* `.streamlit/` – Streamlit configuration and assets
* `pages/` – Modular pages for different analytics views
* `static/` – Images and branding assets for sidebar and interface

---

## 👥 Contributing

We’d love your help! Contributions can include:

* Adding new analytics dashboards
* Improving documentation or tests
* Integrating new data sources
* Enhancing UI/UX design

Before contributing, open an issue to discuss your changes.

---

## 📬 Contact & Resources

If you have questions or want to share your feedback:

📧 Email: **[footbal.data@saulofaria.com.br](mailto:footbal.data@saulofaria.com.br)**
🌐 Website: [https://www.footballhacking.com](https://www.footballhacking.com)
📘 Free E-Books (analytics & betting): linked in the app

Follow the community on social media and YouTube via the sidebar links built into the app.

---

Se quiser, posso também gerar uma **versão em português** ou adicionar badges de CI/CD, cobertura de testes, licença, links de deploy, etc. Quer que eu adicione essas seções?

[1]: https://raw.githubusercontent.com/saulo-data/football.hacking/main/requirements.txt "raw.githubusercontent.com"
