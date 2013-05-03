/*
 * main.cpp
 *
 *  Created on: May 3, 2013
 *      Author: eaglenature@gmail.com
 *
 *  Library features unit testing code.
 */
#include <gcuda/gcuda.h>


TEST(Hello, GCuda)
{
    printf("Hello Steve!\n");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
