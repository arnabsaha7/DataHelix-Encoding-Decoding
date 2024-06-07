# DataHelix: Encoding & Decoding

DataHelix is a Python tool for encoding any dataset into DNA sequences and decoding them back when necessary, aiming to reduce storage space required for large data.

## Installation

To install DataHelix, simply clone this repository and install the required dependencies:

```
git clone https://github.com/arnabsaha7/DataHelix-Encoding-Decoding.git
cd DataHelix-Encoding-Decoding
pip install -r requirements.txt
```

## Usage

To encode a dataset into a DNA sequence, run:

```
python encode_data.py --input dataset.csv --output encoded_sequence.txt
```

To decode a DNA sequence back into the original dataset, run:

```
python decode_data.py --input encoded_sequence.txt --output decoded_dataset.csv
```

## Examples

Here are some examples of encoding and decoding using DataHelix:

### Example

![Encoding Example GIF](examples/encoding.gif)

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

