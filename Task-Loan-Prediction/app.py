from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Form Inputs

        gender = request.form.get("gender")
        married = request.form.get("married")
        education = request.form.get("education")
        self_employed = request.form.get("self_employed")
        property_area = request.form.get("property_area")

        dependents = float(
            request.form.get("dependents", 0)
        )

        applicant_income = float(
            request.form.get("applicant_income", 0)
        )

        coapplicant_income = float(
            request.form.get("coapplicant_income", 0)
        )

        loan_amount = float(
            request.form.get("loan_amount", 0)
        )

        loan_term = float(
            request.form.get("loan_term", 360)
        )

        credit_history = float(
            request.form.get("credit_history", 0)
        )

        # Feature Engineering

        total_income = (
            applicant_income +
            coapplicant_income
        )

        income_per_dependent = (
            total_income /
            (dependents + 1)
        )

        loan_income_ratio = (
            loan_amount /
            (total_income + 1)
        )

        emi = (
            loan_amount /
            (loan_term + 1)
        )

        balance_income = (
            total_income - emi
        )

        loan_amount_log = np.log(
            loan_amount + 1
        )

        total_income_log = np.log(
            total_income + 1
        )

        # Model Input

        input_data = pd.DataFrame([{

            "Dependents":
                dependents,

            "ApplicantIncome":
                applicant_income,

            "CoapplicantIncome":
                coapplicant_income,

            "LoanAmount":
                loan_amount,

            "Loan_Amount_Term":
                loan_term,

            "Credit_History":
                credit_history,

            "TotalIncome":
                total_income,

            "IncomePerDependent":
                income_per_dependent,

            "LoanIncomeRatio":
                loan_income_ratio,

            "EMI":
                emi,

            "BalanceIncome":
                balance_income,

            "LoanAmount_log":
                loan_amount_log,

            "TotalIncome_log":
                total_income_log,

            "Gender_Male":
                1 if gender == "Male" else 0,

            "Married_Yes":
                1 if married == "Yes" else 0,

            "Education_Not Graduate":
                1 if education == "Not Graduate"
                else 0,

            "Self_Employed_Yes":
                1 if self_employed == "Yes"
                else 0,

            "Property_Area_Semiurban":
                1 if property_area == "Semiurban"
                else 0,

            "Property_Area_Urban":
                1 if property_area == "Urban"
                else 0

        }])

        # Prediction

        prediction = model.predict(
            input_data
        )[0]

        # Approval

        if prediction == 1:

            result = "Approved"

            reason = (
                "Strong credit profile, stable income "
                "and acceptable repayment capacity."
            )

        else:

            result = "Rejected"

            reasons = []

            if credit_history == 0:
                reasons.append(
                    "Poor credit history"
                )

            if applicant_income < 2500:
                reasons.append(
                    "Low applicant income"
                )

            if loan_amount > (total_income * 0.8):
                reasons.append(
                    "Loan amount is high compared to income"
                )

            if dependents >= 4:
                reasons.append(
                    "High number of dependents"
                )

            if balance_income < 5000:
                reasons.append(
                    "Insufficient remaining income"
                )

            if loan_term > 480:
                reasons.append(
                    "Long repayment duration"
                )

            if len(reasons) == 0:
                reasons.append(
                    "Profile does not satisfy loan approval criteria"
                )

            reason = reasons

        return render_template(
            "index.html",
            prediction=result,
            reason=reason
        )

    except Exception as e:

        print("ERROR:", e)

        return render_template(
            "index.html",
            prediction="Error",
            reason=[str(e)]
        )


if __name__ == "__main__":
    app.run(debug=True)