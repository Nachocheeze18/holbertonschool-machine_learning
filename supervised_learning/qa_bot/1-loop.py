#!/usr/bin/env python3
"""script that takes in input from the user"""
while True:
  user_input = input("Q:").strip().lower()

  if user_input in ['exit', 'quit', 'goodbye', 'bye']:
      print("A: Goodbye")
      break
  else:
      print("A:")