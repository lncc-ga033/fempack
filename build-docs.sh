#!/bin/bash
# Build fempack documentation with Sphinx (includes automatic API reference)

cd "$(dirname "$0")"

echo "🏗️  Building fempack documentation with Sphinx..."
python -m sphinx -b html docs docs/_build/html "$@"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo "📖 Open docs/_build/html/index.html to view the documentation"
else
    echo ""
    echo "❌ Build failed. Check the output above for errors."
    exit 1
fi
