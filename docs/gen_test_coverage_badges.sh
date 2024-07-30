# CD must be the reports directory!
cd ..
coverage run -m pytest --junitxml=reports/junit/junit.xml test
coverage xml -o  reports/coverage/coverage.xml
coverage html
mv htmlcov/* reports/coverage
rm -r htmlcov

echo "Generating badges"
genbadge tests  -o docs/tests-badge.svg
genbadge coverage  -o docs/coverage-badge.svg
