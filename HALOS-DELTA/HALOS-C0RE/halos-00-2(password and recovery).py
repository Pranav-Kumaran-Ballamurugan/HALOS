# Add to security_tab.py
class AccountRecoverySystem:
    def __init__(self, root):
        self.root = root
        self.setup_recovery_ui()
        self.db = AccountDatabase()  # Would use SQLite/Postgres in production
        
    def setup_recovery_ui(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔑 Account Recovery")
        
        # Recovery Options
        ttk.Label(tab, text="Recover Account Using:").pack(pady=10)
        
        self.recovery_method = tk.StringVar(value="email")
        ttk.Radiobutton(tab, text="Email", variable=self.recovery_method, 
                       value="email").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(tab, text="Facebook", variable=self.recovery_method, 
                       value="facebook").pack(anchor=tk.W, padx=20)
        
        # Input Field
        ttk.Label(tab, text="Registered Email/Facebook ID:").pack(pady=5)
        self.recovery_identifier = ttk.Entry(tab, width=40)
        self.recovery_identifier.pack()
        
        # Security Verification
        ttk.Label(tab, text="Security Question:").pack(pady=5)
        self.security_question = ttk.Label(tab, text="What was your first pet's name?")
        self.security_question.pack()
        
        self.security_answer = ttk.Entry(tab, width=30)
        self.security_answer.pack()
        
        # Action Buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Verify Identity", 
                  command=self.verify_identity).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset Password", 
                  command=self.reset_password, state=tk.DISABLED).pack(side=tk.LEFT)
        
        # Status Display
        self.recovery_status = ttk.Label(tab, text="")
        self.recovery_status.pack()

    def verify_identity(self):
        identifier = self.recovery_identifier.get()
        method = self.recovery_method.get()
        answer = self.security_answer.get()
        
        if not all([identifier, answer]):
            self.recovery_status.config(text="All fields required!", foreground="red")
            return
            
        try:
            # Check against database
            account = self.db.verify_recovery(
                identifier=identifier,
                method=method,
                security_answer=answer
            )
            
            if account:
                self.recovery_status.config(
                    text=f"Verified: {account['username']}", 
                    foreground="green"
                )
                self.enable_password_reset(account)
            else:
                self.recovery_status.config(
                    text="Verification failed", 
                    foreground="red"
                )
                
        except Exception as e:
            self.recovery_status.config(
                text=f"Error: {str(e)}", 
                foreground="red"
            )

    def enable_password_reset(self, account):
        # Enable password reset flow
        self.current_account = account
        self.root.nametowidget(".!notebook.!frame.!button2").config(state=tk.NORMAL)
        
        # Generate and send OTP
        self.otp = str(random.randint(100000, 999999))
        self.send_otp(account)

    def send_otp(self, account):
        # Simulate OTP sending (would use email/sms API in production)
        method = self.recovery_method.get()
        identifier = account['email'] if method == "email" else account['facebook_id']
        
        logging.info(f"OTP {self.otp} sent to {identifier}")
        messagebox.showinfo(
            "Verification Sent", 
            f"OTP sent to your {method} ({identifier[:3]}******)"
        )

    def reset_password(self):
        # OTP verification window
        otp_window = tk.Toplevel(self.root)
        otp_window.title("OTP Verification")
        
        ttk.Label(otp_window, text="Enter 6-digit OTP:").pack(pady=10)
        otp_entry = ttk.Entry(otp_window)
        otp_entry.pack()
        
        def verify_otp():
            if otp_entry.get() == self.otp:
                self.show_password_reset()
                otp_window.destroy()
            else:
                messagebox.showerror("Error", "Invalid OTP")
        
        ttk.Button(otp_window, text="Verify", command=verify_otp).pack(pady=10)

    def show_password_reset(self):
        reset_window = tk.Toplevel(self.root)
        reset_window.title("Reset Password")
        
        ttk.Label(reset_window, text=f"New Password for {self.current_account['username']}").pack(pady=10)
        
        new_pass = ttk.Entry(reset_window, show="•")
        new_pass.pack()
        confirm_pass = ttk.Entry(reset_window, show="•")
        confirm_pass.pack()
        
        def save_password():
            if new_pass.get() != confirm_pass.get():
                messagebox.showerror("Error", "Passwords don't match")
                return
                
            if len(new_pass.get()) < 8:
                messagebox.showerror("Error", "Password must be 8+ characters")
                return
                
            # Update in database
            self.db.update_password(
                username=self.current_account['username'],
                new_password=new_pass.get()  # Would hash in production
            )
            messagebox.showinfo("Success", "Password updated!")
            reset_window.destroy()
        
        ttk.Button(reset_window, text="Save", command=save_password).pack(pady=10)

# Mock Database (would use real DB in production)
class AccountDatabase:
    def __init__(self):
        self.accounts = [
            {
                "username": "user1",
                "email": "user1@example.com",
                "facebook_id": "fb12345",
                "security_answer": "fluffy",
                "password": "oldpass123"  # Would store hashed
            }
        ]
    
    def verify_recovery(self, identifier, method, security_answer):
        for acc in self.accounts:
            if ((method == "email" and acc["email"] == identifier) or
                (method == "facebook" and acc["facebook_id"] == identifier)):
                if acc["security_answer"].lower() == security_answer.lower():
                    return acc
        return None
    
    def update_password(self, username, new_password):
        for acc in self.accounts:
            if acc["username"] == username:
                acc["password"] = new_password  # Hash this in production
                return True
        return False