#include <iostream>
#include <ctime>
#include <iomanip>
#include <omp.h>

using namespace std;

void bubbleSort(int arr[], int n)
{
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

void printArray(int arr[], int n)
{
    for (int i = 0; i < n; ++i)
        cout << arr[i] << " ";
    cout << endl;
}

int main()
{
    int n;
    cout << "Enter the size of the array: ";
    cin >> n;

    int *arr = new int[n];
    int *original = new int[n];

    cout << "Enter " << n << " array elements: ";
    for (int i = 0; i < n; ++i)
    {
        cin >> arr[i];
        original[i] = arr[i];
    }

    cout << "Original array: ";
    printArray(arr, n);

    // Sequential Bubble Sort
    clock_t start = clock();
    bubbleSort(arr, n);
    clock_t end = clock();
    double sequentialTime = double(end - start) / CLOCKS_PER_SEC;

    cout << "Sequential Bubble Sorted array: ";
    printArray(arr, n);

    // Restore original array
    for (int i = 0; i < n; ++i)
        arr[i] = original[i];

    // "Parallel" Bubble Sort (not truly effective)
    start = clock();
    #pragma omp parallel
    {
        bubbleSort(arr, n); // still behaves like sequential
    }
    end = clock();
    double parallelTime = double(end - start) / CLOCKS_PER_SEC;

    cout << "Parallel Bubble Sorted array: ";
    printArray(arr, n);

    cout << fixed << setprecision(8);
    cout << "Sequential Bubble Sort Time: " << sequentialTime << " seconds\n";
    cout << "Parallel Bubble Sort Time:   " << parallelTime << " seconds\n";

    delete[] arr;
    delete[] original;
    return 0;
}
