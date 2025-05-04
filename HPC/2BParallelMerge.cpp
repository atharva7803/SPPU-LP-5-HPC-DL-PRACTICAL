#include <iostream>
#include <ctime>
#include <iomanip>
#include <omp.h>

using namespace std;

void merge(int arr[], int l, int m, int r)
{
    int n1 = m - l + 1;
    int n2 = r - m;

    int *L = new int[n1];
    int *R = new int[n2];

    for (int i = 0; i < n1; ++i)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];

    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

void mergeSort(int arr[], int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(arr, l, m);
            #pragma omp section
            mergeSort(arr, m + 1, r);
        }

        merge(arr, l, m, r);
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

    // Sequential Merge Sort
    clock_t start = clock();
    mergeSort(arr, 0, n - 1);
    clock_t end = clock();
    double sequentialTime = double(end - start) / CLOCKS_PER_SEC;

    cout << "Sequential Merge Sorted array: ";
    printArray(arr, n);

    for (int i = 0; i < n; ++i)
        arr[i] = original[i];

    // Parallel Merge Sort
    start = clock();
    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(arr, 0, n - 1);
    }
    end = clock();
    double parallelTime = double(end - start) / CLOCKS_PER_SEC;

    cout << "Parallel Merge Sorted array: ";
    printArray(arr, n);

    cout << fixed << setprecision(8);
    cout << "Sequential Merge Sort Time: " << sequentialTime << " seconds\n";
    cout << "Parallel Merge Sort Time:   " << parallelTime << " seconds\n";

    delete[] arr;
    delete[] original;
    return 0;
}
