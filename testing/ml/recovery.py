import json
from pathlib import Path
import callback
import os

class Recovery(object):

    def __init__(self, checkpoint, metrics, recover_rate=1.):
        self.checkpoint = checkpoint
        self.recover_rate = recover_rate
        self._path_ = Path.joinpath(checkpoint.checkpoint_path.parent, 'recovery.meta')
        
        item = {
            'acc': 0,
            'epochs': 0,
            'max_epochs': 0,
            'recovery': False
        }
        self._meta_ = item

        if not self._path_.parent.exists():
            raise Exception("not exists recovery path")

        try:
            if self._path_.exists():
                with open(self._path_) as fp:
                    self._meta_ = json.load(fp)
        except:
            pass
        
        self._meta_['recover-epochs'] = max(int(self._meta_['epochs'] * self.recover_rate), 0)
        
        if metrics is not None:
            self.callback = callback.RecoveryCallback(self, metrics)

    def epochs(self):
        if self.is_recovery():
            return self._meta_['recover-epochs']
        return 0
    
    def target(self):
        return self.checkpoint.latest_checkpoint()

    def max_epochs(self, epochs):
        self._meta_['max_epochs'] = epochs
    
    def marking(self, point):
        for key, val in point.items():
            self._meta_[key] = val
        self._meta_['recovery'] = True

    def is_recovery(self):
        return self._meta_['recovery']

    def restore(self, model):
        if self.is_recovery():
            model.load_weights(self.target())
        self._epoch_ = None

    def flush(self):
        self._meta_['recovery'] = False
        self.save()

    def save(self):
        with open(self._path_, 'w') as fp:
            json.dump(self._meta_, fp, indent=4)

    def history(self):
        if self.is_recovery():
            describe = []
            inital_epochs = self._meta_['recover-epochs']
            max_epochs = self._meta_['max_epochs']
            for epochs in range(inital_epochs):
                describe.append(f"Epoch {epochs+1}/{max_epochs} completed. (recovery)\n")
            describe.append(
                "\n"
                "[+] recovery descirbe\n"
                f"--> recover-rate: {int(self._meta_['recover-epochs'] / self._meta_['epochs'] * 100)}%\n"
                f"--> interrupt latest epochs: {self._meta_['recover-epochs']}\n"
                f"--> interrupt target epochs: {self._meta_['epochs']}\n"
            )
            return '\n'.join(describe)
        return "Not found recovery point."

    def interrupt_history(self):
        return '\n'.join([
            f"interrupt path: {self._path_}",
            f"interrupt latest epochs: {self._meta_['recover-epochs']}",
            f"interrupt target epochs: {self._meta_['epochs'] + 1}",
        ])

    def recovery_history(self):
        if self.is_recovery():
            describe = []
            inital_epochs = self._meta_['recover-epochs']
            max_epochs = self._meta_['max_epochs']
            for epochs in range(inital_epochs):
                describe.append(f"Epoch {epochs+1}/{max_epochs} recovery point.\n")
            return '\n'.join(describe)
        return "Not found recovery point."

    def __repr__(self):
        return '\n'.join([
            f"recovery path: {self._path_}",
            f"recovery point target: {self.target()}",
            f"recovery point target exist: {self._meta_['recovery']}",
            f"recovery point target epochs: {self._meta_['epochs']}",
        ])
