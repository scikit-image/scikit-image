.PHONY: logo

logo: green_orange_snake.png snake_logo.svg
	inkscape --export-png=snake_logo.png --export-dpi=100 \
		     --export-area-drawing --export-background-opacity=1 \
			 snake_logo.svg

green_orange_snake.png:
	python scikits_image_logo.py --no-plot

