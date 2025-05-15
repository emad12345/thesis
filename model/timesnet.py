import numpy as np
from pypots.classification.timesnet import TimesNet



n_samples = 1000
n_steps = 50
n_features = 3
n_classes = 3






# 2. ساخت مدل TimesNet
model = TimesNet(
    n_steps=n_steps,
    n_features=n_features,
    n_classes=n_classes,
    n_layers=2,
    top_k=5,
    d_model=64,
    d_ffn=128,
    n_kernels=4,
    dropout=0.1,
    batch_size=32,
    epochs=10,
    verbose=True
)

# 3. آموزش مدل
model.fit(train_set, val_set)

# 4. پیش‌بینی روی داده‌ی جدید (مثلاً از val_set)
test_set = {'X': X_val}

# کلاس‌ها
predictions = model.predict(test_set)
print("Predicted Classes:", predictions['classification'])

# احتمال هر کلاس
probs = model.predict_proba(test_set)
print("Predicted Probabilities:", probs)
