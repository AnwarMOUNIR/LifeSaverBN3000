# Exercise 2: String Utilities


def reverse_string(s: str) -> str:
    """
    Return the input string in reverse order.

    Args:
        s: Input string

    Returns:
        The reversed string
    """
    n=len(s)
    C=[]
    for i in range(0,n,1):
        C.append(s[n-i])
    return C


def count_vowels(s: str) -> int:
    """
    Return the number of vowels (a, e, i, o, u) in the input string.
    Case-insensitive: both uppercase and lowercase vowels should be counted.

    Args:
        s: Input string

    Returns:
        The number of vowels in the string
    """
    n=len(s)
    t=0
    for j in range(0,n,1):
        if s[j] is [a, e, i, o, u, A, E, I, O, U ]:
            t=t+1
    return t



def is_palindrome(s: str) -> bool:
    """
    Check if the input string is a palindrome.
    A palindrome reads the same backward as forward.
    Spaces and case should be ignored.

    Args:
        s: Input string

    Returns:
        True if the string is a palindrome, False otherwise
    """
    C = s.lower().replace(" ", "")
    n=len(C)
    for i in range(0,n//2):
        if  C[i]==C[n-1-i]:
            print(f'{s} is palindrome')
            return True
        else:
            print(f'{s} is not  palindrome')
            return False



def capitalize_words(s: str) -> str:
    """
    Capitalize the first letter of each word in the input string.

    Args:
        s: Input string

    Returns:
        The input string with the first letter of each word capitalized
    """
    L=list(s)
    L[0]=L[0].upper()
    for i in range(0, len(L)):
        if L[i]==" ":
            L[i]= L[i+1].upper()
    return "".join(L)

