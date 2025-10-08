@echo off
echo Renombrando archivos...

REM Archivos principales
ren micmac_complete_app.py app.py
ren requirements_txt.txt requirements.txt
ren readme_github.txt README.md
ren license_file.txt LICENSE
ren changelog_file.txt CHANGELOG.md
ren contributing_guide.txt CONTRIBUTING.md
ren gitignore_file.txt .gitignore
ren quickstart_script.sh quickstart.sh
ren quickstart_windows.txt quickstart.bat
ren resumen_final.txt RESUMEN_PROYECTO.md

REM Archivos en docs
ren docs\metodologia_detallada.txt docs\METODOLOGIA.md
ren docs\casos_uso_practicos.txt docs\CASOS_DE_USO.md
ren docs\faq_document.txt docs\FAQ.md

REM Archivos en .github
ren .github\github_pr_template.txt .github\PULL_REQUEST_TEMPLATE.md
ren .github\workflows\github_ci_workflow.txt .github\workflows\ci.yml

REM Archivos en .github\ISSUE_TEMPLATE
ren .github\ISSUE_TEMPLATE\github_issue_bug.txt .github\ISSUE_TEMPLATE\bug_report.md
ren .github\ISSUE_TEMPLATE\github_issue_feature.txt .github\ISSUE_TEMPLATE\feature_request.md

echo Listo!
pause