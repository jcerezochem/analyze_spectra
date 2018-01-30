
all-local: python-update-version-local

#install-exec-local:
#	cp *.py $(prefix)/bin

python-update-version-local:
	@echo "  "
	@echo "Updating version_tag.py..."
	@echo "Did you first git-commit your changes?"
	@echo "  "
	@./get_git_version.sh || echo "Reusing version file"

python-standalone-local:
	pyinstaller analyze_UVspectrum_pyqt4.py --onefile

