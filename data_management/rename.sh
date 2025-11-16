# Delete mixtures
rm *.0.wav *.3.wav

# Rename singles for clarity
for f in *.1.wav; do
    mv "$f" "${f%.1.wav}.clar.wav"
done
for f in *.2.wav; do
    mv "$f" "${f%.2.wav}.vibes.wav"
done
for f in *.4.wav; do
    mv "$f" "${f%.4.wav}.strings.wav"
done
for f in *.5.wav; do
    mv "$f" "${f%.5.wav}.piano.wav"
done