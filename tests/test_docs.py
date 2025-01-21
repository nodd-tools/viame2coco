import unittest
import utils
import viame2coco.viame_manual_annotations
import viame2coco.viame2coco

def load_tests(loader, tests, ignore):
    modules = [
        viame2coco.viame_manual_annotations,
        viame2coco.viame2coco
    ]
    for module in modules:
        tests = utils.doctests(module, tests)
    return tests

if __name__ == '__main__':
    unittest.main()
