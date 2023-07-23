#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_INPUT_LENGTH 2

int main() {
    char input[MAX_INPUT_LENGTH];

    while (true) {
        printf("Enter 1 to add students to the data base\n");
        printf("Enter 2 to run mark attendance\n");
        printf("Enter q to exit\n");
        printf("Your choice: ");

        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("Error reading input.\n");
            exit(EXIT_FAILURE);
        }

        /* Remove the trailing newline character from fgets */
        size_t len = strlen(input);
        if (len > 0 && input[len - 1] == '\n') {
            input[len - 1] = '\0';
        }

        /* Check user input and execute the corresponding command */
        if (strcmp(input, "1") == 0) {
            system("python3 add_faces.py");
        } else if (strcmp(input, "2") == 0) {
            system("python3 mark.py");
        } else if (strcmp(input, "q") == 0 || strcmp(input, "Q") == 0) {
            printf("Exiting...\n");
            break;
        } else {
            printf("Invalid input. Please try again.\n");
        }
    }

    return 0;
}
