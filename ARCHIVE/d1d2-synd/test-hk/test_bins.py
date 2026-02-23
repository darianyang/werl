def classify_into_bin(bins, value):
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i + 1  # Bin numbers start from 1
    return None  # Return None if the value doesn't fall within any bin

# Example usage:
bins = [1, 3, 6]
value = 5
bin_number = classify_into_bin(bins, value)
print(f"The value {value} belongs to bin {bin_number}.")

