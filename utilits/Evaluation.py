import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class EvaluationReportGenerator:
    def __init__(self, model, log_dir="results"):
        self.model = model
        self.log_dir = log_dir
        self.excel_path = os.path.join(self.log_dir, "full_evaluation_report_colored.xlsx")
        os.makedirs(self.log_dir, exist_ok=True)

    def generate_report(self, X_train, y_train, X_val, y_val, X_test, y_test):
        splits = {
            "train": (X_train, y_train),
            "val":   (X_val, y_val),
            "test":  (X_test, y_test)
        }

        with pd.ExcelWriter(self.excel_path, engine='xlsxwriter') as writer:
            workbook  = writer.book
            worksheet = workbook.add_worksheet("Evaluation")
            writer.sheets["Evaluation"] = worksheet

            # فرمت‌ها
            bold_format = workbook.add_format({'bold': True})
            highlight_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})  # سبز
            red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})       # قرمز

            row = 0  # شروع ردیف

            for split_name, (X, y_true) in splits.items():
                y_pred = self.model.predict({"X": X})["classification"]
                acc = accuracy_score(y_true, y_pred)
                report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                cm = confusion_matrix(y_true, y_pred)
                cm_df = pd.DataFrame(cm)

                # عنوان بخش
                worksheet.write(row, 0, f"{split_name.upper()} EVALUATION", bold_format)
                row += 1

                # دقت
                worksheet.write(row, 0, "Accuracy")
                worksheet.write(row, 1, acc)
                row += 2

                # classification report
                worksheet.write(row, 0, "Classification Report", bold_format)
                row += 1
                report_df.to_excel(writer, sheet_name="Evaluation", startrow=row, startcol=0, index=True)

                # هایلایت بهترین F1-score
                try:
                    f1_scores = report_df.loc[report_df.index.difference(['accuracy', 'macro avg', 'weighted avg']), 'f1-score']
                    best_f1_idx = f1_scores.idxmax()
                    best_f1_row = report_df.index.get_loc(best_f1_idx) + row + 1
                    f1_col = report_df.columns.get_loc('f1-score') + 1
                    worksheet.write(best_f1_row, f1_col, report_df.loc[best_f1_idx, 'f1-score'], highlight_format)
                except Exception as e:
                    print(f"⚠️ Could not highlight F1-score for {split_name}: {e}")

                row += len(report_df) + 3

                # confusion matrix
                worksheet.write(row, 0, "Confusion Matrix", bold_format)
                row += 1
                cm_df.to_excel(writer, sheet_name="Evaluation", startrow=row, startcol=0, index=False, header=False)

                # هایلایت بزرگ‌ترین مقدار
                max_val = cm_df.values.max()
                for i in range(cm_df.shape[0]):
                    for j in range(cm_df.shape[1]):
                        cell_val = cm_df.iat[i, j]
                        cell_row = row + i
                        cell_col = j
                        fmt = highlight_format if cell_val == max_val else None
                        worksheet.write(cell_row, cell_col, cell_val, fmt)

                row += cm_df.shape[0] + 4

        print(f"saved report {self.excel_path}")
