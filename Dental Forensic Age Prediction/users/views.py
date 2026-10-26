# USER VIEWS
from django.shortcuts import render, redirect
from django.contrib import messages
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
from users.forms import UserRegistrationForm
from users.models import UserRegistrationModel

def UserLogin(request):
    return render(request, 'UserLogin.html')

def user_login_required(view_func):
    def wrapper(request, *args, **kwargs):
        if 'loggeduser' not in request.session:
            return redirect('please_login')  # redirect to your custom login page
        return view_func(request, *args, **kwargs)
    return wrapper

def UserRegisterForm(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrationForm.html', {'form': form})


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered!')
            return redirect('AdminLogin')
        else:
            messages.error(request, 'Email or Mobile already exists')
    else:
        form = UserRegistrationForm()

    return render(request, 'UserRegistrationForm.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        password = request.POST.get('password')

        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=password)

            request.session['loggeduser'] = user.Username
            request.session['loginid'] = user.loginid
            request.session['email'] = user.email
            request.session['last_training_time'] = None
            request.session['last_model_accuracy'] = None
            request.session['last_model_name'] = None
            request.session['last_correct_preds'] = None
            request.session['last_incorrect_preds'] = None
            request.session['last_prediction_time'] = None

            print("✅ Session set successfully:", request.session.items()) 
            return redirect('Userbase')

        except UserRegistrationModel.DoesNotExist:
            messages.error(request, "Invalid Login ID or Password")
            return redirect('UserLogin')

    return redirect('UserLogin')

@user_login_required
def Userbase(request):
    return render(request, 'users/Userhome.html')


import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

import tensorflow as tf


APP_DIR = os.path.dirname(__file__)
IMG_SIZE = 224


# ---------------------------------------------------------
# LOGIN DECORATOR
# ---------------------------------------------------------
def user_login_required(view_func):
    def wrapper(request, *args, **kwargs):
        if 'loggeduser' not in request.session:
            from django.shortcuts import redirect
            return redirect('UserLogin')
        return view_func(request, *args, **kwargs)
    return wrapper


# ---------------------------------------------------------
# MODEL + METADATA PATHS
# ---------------------------------------------------------
MODEL_PATH = os.path.join(settings.MEDIA_ROOT, "fine_tuned_forensic_under12_model.keras")
CLASS_INDEX_PATH = os.path.join(settings.MEDIA_ROOT, "class_indices.json")
THRESHOLD_PATH = os.path.join(settings.MEDIA_ROOT, "best_threshold.json")

# Lazy loading for model and metadata
_model = None
_class_indices = None
_best_threshold = None
_idx_to_class = None

def get_model():
    global _model
    if _model is None:
        try:
            # Try to load the model with custom objects to handle version compatibility
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            # If loading fails, try to recreate the model architecture and load weights
            try:
                # Create EfficientNetB0 base model
                base_model = tf.keras.applications.EfficientNetB0(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg'
                )
                
                # Freeze the base model
                base_model.trainable = False
                
                # Add custom layers
                inputs = base_model.input
                x = tf.keras.layers.BatchNormalization()(base_model.output)
                x = tf.keras.layers.Dropout(0.35)(x)
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                
                # Create the model
                _model = tf.keras.Model(inputs, outputs)
                
                # Try to load weights if the model file contains them
                try:
                    _model.load_weights(MODEL_PATH)
                    print("Model weights loaded successfully")
                except:
                    print("Could not load weights, using base model only")
                    
            except Exception as e2:
                print(f"Error creating model: {e2}")
                _model = None
                raise e
    return _model

def get_class_indices():
    global _class_indices
    if _class_indices is None:
        with open(CLASS_INDEX_PATH, "r") as f:
            _class_indices = json.load(f)
    return _class_indices

def get_best_threshold():
    global _best_threshold
    if _best_threshold is None:
        with open(THRESHOLD_PATH, "r") as f:
            _best_threshold = float(json.load(f)["best_threshold"])
    return _best_threshold

def get_idx_to_class():
    global _idx_to_class
    if _idx_to_class is None:
        _idx_to_class = {v: k for k, v in get_class_indices().items()}
    return _idx_to_class


# ---------------------------------------------------------
# PREPROCESS FUNCTION
# ---------------------------------------------------------
def _preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr


# ---------------------------------------------------------
# TRAINING RESULTS VIEW
# ---------------------------------------------------------

import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import render

@user_login_required
def Training(request):
    history_path = os.path.join(settings.MEDIA_ROOT, "training_history.json")
    pred_path = os.path.join(settings.MEDIA_ROOT, "sample_predictions.csv")
    report_path = os.path.join(settings.MEDIA_ROOT, "classification_report.csv")

    context = {}

    plot_dir = os.path.join(settings.MEDIA_ROOT, "training_plots")
    os.makedirs(plot_dir, exist_ok=True)

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            hist = json.load(f)

        acc = hist.get("accuracy", [])
        val_acc = hist.get("val_accuracy", [])

        loss = hist.get("loss", [])
        val_loss = hist.get("val_loss", [])

        auc = hist.get("auc", [])
        val_auc = hist.get("val_auc", [])

        precision = hist.get("precision", [])
        val_precision = hist.get("val_precision", [])

        recall = hist.get("recall", [])
        val_recall = hist.get("val_recall", [])

        # fallback in case some values are None
        acc = acc or []
        val_acc = val_acc or []
        loss = loss or []
        val_loss = val_loss or []
        auc = auc or []
        val_auc = val_auc or []
        precision = precision or []
        val_precision = val_precision or []
        recall = recall or []
        val_recall = val_recall or []

        # choose epoch length from first available metric
        metric_lengths = [
            len(acc), len(val_acc), len(loss), len(val_loss),
            len(auc), len(val_auc), len(precision), len(val_precision),
            len(recall), len(val_recall)
        ]
        max_len = max(metric_lengths) if metric_lengths else 0
        epochs = range(1, max_len + 1)

        # LOSS PLOT
        if loss or val_loss:
            loss_file = os.path.join(plot_dir, "loss_curve.png")
            plt.figure()
            if loss:
                plt.plot(range(1, len(loss) + 1), loss, label="Train Loss")
            if val_loss:
                plt.plot(range(1, len(val_loss) + 1), val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(loss_file)
            plt.close()
            context["loss_plot"] = settings.MEDIA_URL + "training_plots/loss_curve.png"

        # ACCURACY PLOT
        if acc or val_acc:
            acc_file = os.path.join(plot_dir, "accuracy_curve.png")
            plt.figure()
            if acc:
                plt.plot(range(1, len(acc) + 1), acc, label="Train Accuracy")
            if val_acc:
                plt.plot(range(1, len(val_acc) + 1), val_acc, label="Val Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(acc_file)
            plt.close()
            context["accuracy_plot"] = settings.MEDIA_URL + "training_plots/accuracy_curve.png"

        # AUC PLOT
        if auc or val_auc:
            auc_file = os.path.join(plot_dir, "auc_curve.png")
            plt.figure()
            if auc:
                plt.plot(range(1, len(auc) + 1), auc, label="Train AUC")
            if val_auc:
                plt.plot(range(1, len(val_auc) + 1), val_auc, label="Val AUC")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.legend()
            plt.tight_layout()
            plt.savefig(auc_file)
            plt.close()
            context["auc_plot"] = settings.MEDIA_URL + "training_plots/auc_curve.png"

        # PRECISION PLOT
        if precision or val_precision:
            precision_file = os.path.join(plot_dir, "precision_curve.png")
            plt.figure()
            if precision:
                plt.plot(range(1, len(precision) + 1), precision, label="Train Precision")
            if val_precision:
                plt.plot(range(1, len(val_precision) + 1), val_precision, label="Val Precision")
            plt.xlabel("Epoch")
            plt.ylabel("Precision")
            plt.legend()
            plt.tight_layout()
            plt.savefig(precision_file)
            plt.close()
            context["precision_plot"] = settings.MEDIA_URL + "training_plots/precision_curve.png"

        # RECALL PLOT
        if recall or val_recall:
            recall_file = os.path.join(plot_dir, "recall_curve.png")
            plt.figure()
            if recall:
                plt.plot(range(1, len(recall) + 1), recall, label="Train Recall")
            if val_recall:
                plt.plot(range(1, len(val_recall) + 1), val_recall, label="Val Recall")
            plt.xlabel("Epoch")
            plt.ylabel("Recall")
            plt.legend()
            plt.tight_layout()
            plt.savefig(recall_file)
            plt.close()
            context["recall_plot"] = settings.MEDIA_URL + "training_plots/recall_curve.png"

        # final metrics
        if val_acc:
            last_val_accuracy = round(float(val_acc[-1]), 3)
            best_val_accuracy = round(float(max(val_acc)), 3)
            context["last_val_accuracy"] = last_val_accuracy
            context["best_val_accuracy"] = best_val_accuracy
            context["last_val_accuracy_pct"] = round(last_val_accuracy * 100, 1)
            context["best_val_accuracy_pct"] = round(best_val_accuracy * 100, 1)
        if val_auc:
            last_val_auc = round(float(val_auc[-1]), 3)
            context["last_val_auc"] = last_val_auc
            context["last_val_auc_pct"] = round(last_val_auc * 100, 1)

    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path)
        context["pred_table"] = df.to_html(index=False, classes="table table-striped table-bordered")



    # With this:
    if os.path.exists(report_path):
        df_report = pd.read_csv(report_path, index_col=0)
        if "accuracy" in df_report.index and "precision" in df_report.columns:
            test_accuracy = round(float(df_report.loc["accuracy", "precision"]), 3)
            context["test_accuracy"] = test_accuracy
            context["test_accuracy_pct"] = round(test_accuracy * 100, 1)
        df_report.index.name = "Class"
        df_report = df_report.reset_index()

        # Clean column names: f1-score → F1 SCORE etc.
        df_report.columns = [c.replace("-", " ").upper() for c in df_report.columns]

        # Fix sklearn accuracy row: it places score in 'precision' and support in 'recall'
        if "CLASS" in df_report.columns and "PRECISION" in df_report.columns:
            acc_mask = df_report["CLASS"].astype(str).str.strip().str.lower() == "accuracy"
            if acc_mask.any():
                df_report.loc[acc_mask, "F1 SCORE"] = df_report.loc[acc_mask, "PRECISION"].values
                df_report.loc[acc_mask, "SUPPORT"] = df_report.loc[acc_mask, "RECALL"].values
                df_report.loc[acc_mask, ["PRECISION", "RECALL"]] = "—"

        # Round numeric columns
        for col in df_report.select_dtypes(include="number").columns:
            df_report[col] = df_report[col].round(4)

        context["report_table"] = df_report.to_html(
            index=False,
            classes="table table-striped table-bordered",
            escape=False,
            na_rep="—"
        )

    return render(request, "users/training_results.html", context)


# ---------------------------------------------------------
# PREDICTION VIEW
# ---------------------------------------------------------
@user_login_required
def PredictTeethAge(request):
    context = {}

    if request.method == "POST" and request.FILES.get("xray"):
        xray = request.FILES["xray"]

        upload_dir = os.path.join(settings.MEDIA_ROOT, "xray_uploads")
        os.makedirs(upload_dir, exist_ok=True)

        fs = FileSystemStorage(location=upload_dir)
        filename = fs.save(xray.name, xray)
        file_path = fs.path(filename)
        file_url = settings.MEDIA_URL + "xray_uploads/" + filename

        try:
            x = _preprocess(file_path)

            model = get_model()
            prob = float(model.predict(x, verbose=0)[0][0])
            pred_idx = int(prob >= get_best_threshold())
            pred_class = get_idx_to_class()[pred_idx]

            below_12_probability = prob
            age_12_and_above_probability = 1.0 - prob

            if pred_class == "below_12":
                result_label = "12 and Below"
                interpretation = "The model predicts that the dental X-ray belongs to a person aged 12 years or below."
            else:
                result_label = "12 and Above"
                interpretation = "The model predicts that the dental X-ray belongs to a person aged 12 years or above."

            context = {
                "file_url": file_url,
                "predicted_class": pred_class,
                "result_label": result_label,
                "probability": round(prob, 4),
                "below_12_probability": round(below_12_probability, 4),
                "age_12_and_above_probability": round(age_12_and_above_probability, 4),
                "threshold": round(get_best_threshold(), 4),
                "interpretation": interpretation,
            }

        except Exception as e:
            context["error"] = f"Prediction failed: {str(e)}"

    return render(request, "users/predict_teeth_age.html", context)