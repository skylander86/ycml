from argparse import ArgumentParser
from secrets import token_bytes, choice, token_hex
import string


def main():
    parser = ArgumentParser(description='Generate a secret key.')
    parser.add_argument('-l', '--length', type=int, default=32, help='Length of secret key in bytes.')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-x', '--hex', action='store_true', help='Convert secret key to hexadecimal.')
    group.add_argument('-a', '--alphanum', action='store_true', help='Generate alphanumeric keys only.')
    A = parser.parse_args()

    if A.alphanum:
        alphabet = string.ascii_letters + string.digits
        print(''.join(choice(alphabet) for i in range(A.length)))
    elif A.hex: print(token_hex(A.length))
    else: print(token_bytes(A.length))
    #end if
#end def


if __name__ == '__main__': main()
