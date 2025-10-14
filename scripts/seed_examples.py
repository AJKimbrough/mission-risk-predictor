from app.etl.nws_grid import seed_dummy


if __name__ == "__main__":
    seed_dummy(32.7767, -96.7970)
    print("Seeded 24h dummy data for Dallas (approx).")