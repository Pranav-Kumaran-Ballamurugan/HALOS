class HALOSGUI:
    def __init__(self, core: HALOSCore):
        self.core = core
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QMainWindow()
        
        # Setup tabs
        self.tabs = QtWidgets.QTabWidget()
        self._setup_llm_tab()
        self._setup_security_tab()
        self._setup_code_tab()
        # ... other tabs
        
        # Dark/light mode
        self.dark_mode = False
        self._setup_theme_toggle()
        
        # Notification center
        self.notifications = NotificationCenter()
    
    def _setup_theme_toggle(self):
        theme_btn = QtWidgets.QPushButton("Toggle Theme")
        theme_btn.clicked.connect(self.toggle_theme)
    
    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self._apply_theme()