
********
There are four collection data types in the Python programming language:

**List** is a collection which is **ordered** and **changeable**. Allows duplicate members.

**Tuple** is a collection which is **ordered** and **unchangeable**. Allows duplicate members.

**Set** is a collection which is **unordered** and **unchangeable**, and unindexed. No duplicate members.

**Dictionary** is a collection which is **ordered** and **changeable**. No duplicate members.

**Mutable - changeable**

**Immutable - non changeable**

******

# Lists ( Mutable )
mylist = ["apple", "banana", "cherry"]

Lists are used to store multiple items in a single variable.

Lists are one of 4 built-in data types in Python used to store collections of data, the other 3 are Tuple, Set, and Dictionary, all with different qualities and usage.

**Lists are created using square brackets:**



       fruits = ["apple", "banana", "cherry", "kiwi", "mango"]
       newlist = []

       for x in fruits:
          if "a" in x:
          newlist.append(x)

       print(newlist)

#### 1. How to add/join 2 lists

       list1 = ["a", "b", "c"]
       list2 = [1, 2, 3]
       list3 = list1 + list2
       
  **OR**
       
       for x in list2:
        list1.append(x)

  **OR**

  Use the extend() method to add list2 at the end of list1:
       
       list1.extend(list2)
#### 2. Remove the second item:

     thislist = ["apple", "banana", "cherry"]
     thislist.pop(1)
     print(thislist)

#### 3. Adv features 

    fruits = ["apple", "banana", "cherry", "kiwi", "mango"]
    newlist = [x.upper() for x in fruits]
    print(newlist)

    &&
    
    newlist1 = [x if x != "banana" else "orange" for x in fruits]

#### 4. Sort

    thislist = [100, 50, 65, 82, 23]
    thislist.sort(reverse = True) #sort in descending order
*********

# TUPLES ( IMMUTABLE )

     tuple1 = ("abc", 34, True, 40, "male")


Print the last item of the tuple: 

     print(thistuple[-1])

    thistuple = ("apple", "banana", "cherry")
    if "apple" in thistuple:
      print("Yes, 'apple' is in the fruits tuple")


#### 1. Change Tuple Values
Once a tuple is created, you cannot change its values. Tuples are unchangeable, or immutable as it also is called.

But there is a workaround. You can convert the tuple into a list, change the list, and convert the list back into a tuple.


    x = ("apple", "banana", "cherry")
    y = list(x)
    y[1] = "kiwi"
    x = tuple(y)

    *** ("apple", "kiwi", "cherry") ***


#### 2. Add tuple to a tuple.
You are allowed to add tuples to tuples, so if you want to add one item, (or many), create a new tuple with the item(s), and add it to the existing tuple:

     thistuple = ("apple", "banana", "cherry")
     y = ("orange",)
     thistuple += y

#### 3. Unpacking a Tuple
To extract the values back from tuple into variables. This is called "unpacking":

     fruits = ("apple", "banana", "cherry")
     (var1,var2,var3) = fruits



If the number of variables is less than the number of values, you can add an * to the variable name and the values will be assigned to the variable as a list:

     fruits = ("apple", "mango", "papaya", "pineapple", "cherry")
     (green, *tropic, red) = fruits

#### 4. Multiply content
If you want to multiply the content of a tuple a given number of times, you can use the * operator:

    fruits = ("apple", "banana", "cherry")
    mytuple = fruits * 2
    
    mytuple = ("apple", "banana", "cherry","apple", "banana", "cherry")

*********

# SETS ( IMMUTABLE )


**Note: Set items are unchangeable, but you can remove items and add new items.**

    thisset = {"apple", "banana", "cherry"}
    print(thisset)

    thisset = {"apple", "banana", "cherry"}
    print("banana" in thisset)
    ## OUTPUT : TRUE

#### 1. To add one item to a set use the add() method.
#### 2. To add items from another set into the current set, use the update() method.
#### 3. To remove an item in a set, use the remove(), or the discard() method.
#### 4. You can also use the pop() method to remove an item, but this method will remove a random item, so you cannot be sure what item that gets removed.
#### 5. UNION 
    set1 = {"a", "b" , "c"}
    set2 = {1, 2, 3}
    set3 = set1.union(set2)
#### 6. INTERSECTION


    x = {"apple", "banana", "cherry"}
    y = {"google", "microsoft", "apple"}
    x.intersection_update(y)

#### 7. Return a set that contains all items from both sets, except items that are present in both:

    x = {"apple", "banana", "cherry"}
    y = {"google", "microsoft", "apple"}
    z = x.symmetric_difference(y)

<img width="1440" alt="Screenshot 2023-08-15 at 9 46 43 PM" src="https://github.com/TirthGada/MachineLearning_practice/assets/118129263/a9171c48-264b-4082-b17f-5e6b10628ad7">
<img width="1440" alt="Screenshot 2023-08-15 at 11 24 13 PM" src="https://github.com/TirthGada/MachineLearning_practice/assets/118129263/90181693-5b3b-425e-973e-b8f6d28580b1">
<img width="1440" alt="Screenshot 2023-08-15 at 9 44 26 PM" src="https://github.com/TirthGada/MachineLearning_practice/assets/118129263/59bd40ab-543d-4cdb-9084-01356527f882">

