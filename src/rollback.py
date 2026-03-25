class ModelManager:
    def __init__(self, stable_model, primary_model=None):
        self.stable_model = stable_model
        self.primary_model = primary_model or stable_model

        self.current_model = self.primary_model
        self.rollback_active = False

        # Recovery tracking
        self.recovery_counter = 0
        self.recovery_required_windows = 5

    def rollback(self):
        self.current_model = self.stable_model
        self.rollback_active = True
        self.recovery_counter = 0

    def try_recover(self, system_stable: bool):
        """
        Recover only if system is stable for several consecutive windows
        """
        if not self.rollback_active:
            return False

        if system_stable:
            self.recovery_counter += 1
        else:
            self.recovery_counter = 0

        if self.recovery_counter >= self.recovery_required_windows:
            self.current_model = self.primary_model
            self.rollback_active = False
            self.recovery_counter = 0
            return True

        return False

    def predict(self, X):
        return self.current_model.predict(X)

    def status(self):
        return "rollback_active" if self.rollback_active else "normal"
