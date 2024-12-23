import streamlit as st
import json
from fpdf import FPDF

# File path for storing reports
REPORTS_FILE = "reports.json"

# Load existing reports from file
def load_reports(file_path=REPORTS_FILE):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# Function to generate PDF
def generate_pdf(reports):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Safety Compliance Reports", ln=True, align='C')
    pdf.ln(10)  # Add a line break

    for report in reports:
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, txt=f"ID: {report['id']}", ln=True)
        pdf.cell(0, 10, txt=f"Date and Time: {report['date_time']}", ln=True)
        pdf.cell(0, 10, txt=f"Status: {report['status']}", ln=True)
        missing_items = ", ".join(report["missing_items"]) if report["missing_items"] else "None"
        pdf.cell(0, 10, txt=f"Missing Items: {missing_items}", ln=True)
        pdf.ln(5)  # Add space between reports

    # Add summary statistics
    total_reports = len(reports)
    compliant = len([r for r in reports if r["status"] == "Compliant"])
    non_compliant = total_reports - compliant

    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Summary Statistics", ln=True, align='L')
    pdf.ln(5)
    pdf.cell(0, 10, txt=f"Total Reports: {total_reports}", ln=True)
    pdf.cell(0, 10, txt=f"Compliant Individuals: {compliant}", ln=True)
    pdf.cell(0, 10, txt=f"Non-Compliant Individuals: {non_compliant}", ln=True)

    return pdf

# View and print reports page
def main():
    st.title("View Reports")

    # Load reports
    reports = load_reports()

    if reports:
        # Display each report
        for report in reports:
            st.write(f"**ID:** {report['id']}")
            st.write(f"**Date and Time:** {report['date_time']}")
            st.write(f"**Status:** {report['status']}")
            st.write(f"**Missing Items:** {', '.join(report['missing_items']) if report['missing_items'] else 'None'}")
            st.write("---")

        # Summary statistics
        st.header("Summary Statistics")
        total_reports = len(reports)
        compliant = len([r for r in reports if r["status"] == "Compliant"])
        non_compliant = total_reports - compliant

        st.write(f"**Total Reports:** {total_reports}")
        st.write(f"**Compliant Individuals:** {compliant}")
        st.write(f"**Non-Compliant Individuals:** {non_compliant}")

        # Generate and download PDF
        if st.button("Download PDF Report"):
            pdf = generate_pdf(reports)
            pdf_output = "reports_summary.pdf"
            pdf.output(pdf_output)
            with open(pdf_output, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF",
                    data=pdf_file,
                    file_name=pdf_output,
                    mime="application/pdf"
                )
    else:
        st.write("No reports available.")

if __name__ == "__main__":
    main()
