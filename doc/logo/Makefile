.PHONY: logo

logo: green_orange_snake.png snake_logo.svg
	inkscape --export-png=scikit_image_logo.png --export-dpi=100 \
		     --export-area-drawing --export-background-opacity=1 \
			 snake_logo.svg
	python shrink_logo.py

green_orange_snake.png:
	python scikit_image_logo.py --no-plot

