APP_STYLESHEET = """
QWidget {
    background-color: #eef5fb;
    color: #17324d;
    font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #e9f2fb;
}

QFrame#Card {
    background-color: #ffffff;
    border: 1px solid #c9dff4;
    border-radius: 14px;
}

QLabel#Title {
    font-size: 30px;
    font-weight: 700;
    color: #1d486b;
}

QLabel#Subtitle {
    font-size: 15px;
    color: #335f82;
}

QProgressBar {
    border: 1px solid #9bbdd9;
    border-radius: 8px;
    text-align: centre;
    background: #f5f9fd;
    min-height: 18px;
}

QProgressBar::chunk {
    background-color: #4a9bd6;
    border-radius: 6px;
}

QPushButton {
    background-color: #2f8ac9;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #2576ae;
}

QPushButton:disabled {
    background-color: #a5bfd3;
    color: #eaf2f8;
}

QListWidget, QTableWidget {
    border: 1px solid #b8d1e6;
    border-radius: 8px;
    background: #ffffff;
}

QHeaderView::section {
    background-color: #d8e9f7;
    color: #1d486b;
    border: none;
    padding: 6px;
    font-weight: 600;
}
"""
