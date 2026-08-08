install:
	uv sync

run:
	uv run python -m streamlit run main.py

test_dummy_dataset:
	uv run python src_data_readyness_agent/testing/test_dummy_data.py

test_car_prices_dataset:
	uv run python src_data_readyness_agent/testing/test_car_data.py

test_movie_ratings_dataset:
	uv run python src_data_readyness_agent/testing/test_movies_data.py