echo "The iTeamTests..."
echo "#"
FILES=./iTeam_test/*.py
for f in $FILES
do
        echo ""
        echo "----- TEST: $f -----"
        echo ""
        python py_sym.py eval "$f"

        echo ""
        read -p "--> Press enter to run next test" yn
done
echo "#######################end of iTeamTests"
