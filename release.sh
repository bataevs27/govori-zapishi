#!/bin/bash
# Публикация нового релиза Stenograf
# Использование: ./release.sh 1.1 "Что изменилось в этой версии"

set -e

VERSION="$1"
NOTES="$2"

if [ -z "$VERSION" ] || [ -z "$NOTES" ]; then
    echo "Использование: ./release.sh <версия> <описание>"
    echo "Пример: ./release.sh 1.2 \"Исправлен баг с токеном, добавлены иконки\""
    exit 1
fi

TAG="v$VERSION"

echo "→ Синхронизирую скрипты в бандл..."
cp menubar.py GovoriZapishi.app/Contents/Resources/menubar.py
cp settings_window.py GovoriZapishi.app/Contents/Resources/settings_window.py
cp menubar.py distr/GovoriZapishi.app/Contents/Resources/menubar.py
cp settings_window.py distr/GovoriZapishi.app/Contents/Resources/settings_window.py

echo "→ Синхронизирую гайд в distr..."
cp GUIDE.md distr/GUIDE.md

echo "→ Обновляю версию в Info.plist..."
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" GovoriZapishi.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" distr/GovoriZapishi.app/Contents/Info.plist

echo "→ Создаю архив для релиза..."
cd distr
zip -r "../release_${TAG}.zip" GovoriZapishi.app GUIDE.md
cd ..

echo "→ Коммит и тег..."
git add .
git commit -m "Release $TAG: $NOTES"
git tag -a "$TAG" -m "$NOTES"
git push origin main --tags

echo "→ Публикую релиз на GitHub..."
gh release create "$TAG" \
    "release_${TAG}.zip#Stenograf ${TAG} (macOS Apple Silicon)" \
    --title "Stenograf $TAG" \
    --notes "$NOTES" \
    --latest

rm "release_${TAG}.zip"

echo ""
echo "✅ Релиз $TAG опубликован!"
echo "   https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/tag/$TAG"
