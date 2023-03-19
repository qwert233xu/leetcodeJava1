package com.xu;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Unit test for simple App.
 */
public class AppTest {
    /**
     * Rigorous Test :-)
     */
    @Test
    public void shouldAnswerWithTrue() {
        assertTrue(true);
    }

    @Test
    public void testReverseList() {
        App app = new App();
        ListNode root = new ListNode(1);
        root.next = new ListNode(2);
        root.next.next = new ListNode(3);
        ListNode temp = app.ReverseList3(root);
        while (temp != null) {
            System.out.println(temp.val);
            temp = temp.next;
        }
    }

    @Test
    public void testReverseBetween() {
        App app = new App();
        ListNode root = new ListNode(1);
        root.next = new ListNode(2);
        root.next.next = new ListNode(3);
        root.next.next.next = new ListNode(4);
        root.next.next.next.next = new ListNode(5);
        int m = 2;
        int n = 4;
        ListNode temp = app.reverseBetween2(root, m, n);
        while (temp != null) {
            System.out.println(temp.val);
            temp = temp.next;
        }
    }

    @Test
    public void testReverseKGroup() {
        App app = new App();
        ListNode root = new ListNode(1);
        root.next = new ListNode(2);
        root.next.next = new ListNode(3);
        root.next.next.next = new ListNode(4);
//        root.next.next.next.next = new ListNode(5);
        ListNode res = app.reverseKGroup(root, 2);
        while (res != null) {
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    public void testMerge() {
        App app = new App();
        ListNode list1 = new ListNode(1);
        list1.next = new ListNode(3);
        list1.next.next = new ListNode(5);
        ListNode list2 = new ListNode(2);
        list2.next = new ListNode(4);
        list2.next.next = new ListNode(6);
        ListNode res = app.Merge(list1, list2);
        while (res != null) {
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    public void testDivide() {
        App app = new App();
        ArrayList<ListNode> res = new ArrayList<>();
        res.add(new ListNode(1));
        res.add(new ListNode(2));
        res.add(new ListNode(1));
        res.add(new ListNode(4));
        res.add(new ListNode(5));
        res.add(new ListNode(6));
        ListNode result = app.mergeKLists(res);
        while (result != null) {
            System.out.println(result.val);
            result = result.next;
        }
    }

    @Test
    public void testHasCycle() {
        App app = new App();
        ListNode root = new ListNode(3);
        root.next = new ListNode(2);
        root.next.next = new ListNode(0);
        root.next.next.next = new ListNode(-4);
        root.next.next.next.next = root.next;
        boolean res = app.hasCycle(root);
        System.out.println(res);
    }

    @Test
    public void testEntryNodeOfLoop() {
        App app = new App();
        ListNode root = new ListNode(1);
        root.next = new ListNode(2);
        root.next.next = new ListNode(3);
        root.next.next.next = new ListNode(4);
        root.next.next.next.next = new ListNode(5);
        root.next.next.next.next.next = root.next.next;

        ListNode res = app.EntryNodeOfLoop(root);
        System.out.println(res.val);
    }

    @Test
    public void testFindKthToTail() {
        App app = new App();
        ListNode root = new ListNode(1);
        root.next = new ListNode(2);
        root.next.next = new ListNode(3);
        root.next.next.next = new ListNode(4);
        root.next.next.next.next = new ListNode(5);
        ListNode res = app.FindKthToTail(root, 2);
        while (res != null) {
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    public void testRemoveNthFromEnd() {
        App app = new App();
        ListNode root = new ListNode(1);
        root.next = new ListNode(2);
        ListNode res = app.removeNthFromEnd(root, 2);
        while (res != null) {
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    public void testFindFirstCommonNode() {

        App app = new App();
        ListNode pHead1 = new ListNode(1);
        pHead1.next = new ListNode(2);
        pHead1.next.next = new ListNode(3);
        pHead1.next.next.next = new ListNode(6);
        pHead1.next.next.next.next = new ListNode(7);

        ListNode pHead2 = new ListNode(4);
        pHead2.next = new ListNode(5);
        pHead2.next.next = pHead1.next.next.next;
        pHead2.next.next.next = pHead1.next.next.next.next;

        ListNode res = app.FindFirstCommonNode(pHead1, pHead2);
        while (res != null) {
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    public void testAddInList() {
        App app = new App();
        ListNode pHead1 = new ListNode(0);

        ListNode pHead2 = new ListNode(6);
        pHead2.next = new ListNode(3);

        ListNode res = app.addInList(pHead1, pHead2);
        while (res != null) {
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    public void testSortInList() {
        App app = new App();
        ListNode head = new ListNode(1);
        head.next = new ListNode(3);
        head.next.next = new ListNode(2);
        head.next.next.next = new ListNode(4);
        head.next.next.next.next = new ListNode(5);
        ListNode res = app.sortInList(head);
        while (res != null) {
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    public void testIsPail() {
        App app = new App();
        ListNode head = new ListNode(1);
        head.next = new ListNode(2);
        head.next.next = new ListNode(3);
        head.next.next.next = new ListNode(4);
        head.next.next.next.next = new ListNode(5);
        head.next.next.next.next.next = new ListNode(4);
        head.next.next.next.next.next.next = new ListNode(3);
        head.next.next.next.next.next.next.next = new ListNode(2);
        head.next.next.next.next.next.next.next.next = new ListNode(1);
        head.next.next.next.next.next.next.next.next.next = new ListNode(1);
        boolean pail = app.isPail(head);
        System.out.println(pail);
    }

    @Test
    public void testOddEvenList() {
        App app = new App();
        ListNode root = new ListNode(1);
        root.next = new ListNode(2);
        root.next.next = new ListNode(3);
        root.next.next.next = new ListNode(4);
        root.next.next.next.next = new ListNode(5);
        root.next.next.next.next.next = new ListNode(6);

        ListNode res = app.oddEvenList(root);
        while (res != null) {
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    public void testDeleteDuplicates() {
        App app = new App();
        ListNode head = new ListNode(1);
        head.next = new ListNode(1);
        ListNode res = app.deleteDuplicates(head);
        while (res != null) {
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    public void testDeleteDuplicates2() {
        App app = new App();
        ListNode head = new ListNode(1);
        head.next = new ListNode(2);
        head.next.next = new ListNode(2);
        ListNode res = app.deleteDuplicates2(head);
        while (res != null) {
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    public void testSearch() {
        App app = new App();
//        System.out.println(app.search(new int[]{-1, 0, 3, 4, 6, 10, 13, 14}, 13));
        System.out.println(app.search(new int[]{-1, 1}, -1));
    }

    @Test
    public void testFindPeakElement() {
        App app = new App();
        int peakElement = app.findPeakElement(new int[]{2, 4, 1, 2, 7, 8, 4});
        System.out.println(peakElement);
    }

    @Test
    public void testInversePairs() {
        App app = new App();
        System.out.println(app.InversePairs(new int[]{1, 2, 3, 4, 5, 6, 7, 0}));
    }

    @Test
    public void testPriorityQueueSort() {
        App app = new App();
        System.out.println(Arrays.toString(app.PriorityQueueSort(new int[]{3, 2, 5, 1, 7})));
    }

    @Test
    public void testCompare() {
        App app = new App();
        System.out.println(app.compare("1.1", "2.1"));
    }

    @Test
    public void testLevelOrder() {
        App app = new App();
        TreeNode root = new TreeNode(3);
        root.left = new TreeNode(9);
        root.right = new TreeNode(20);
        root.right.left = new TreeNode(15);
        root.right.right = new TreeNode(7);
        System.out.println(app.levelOrder(root));
    }

    @Test
    public void testPrint() {
        App app = new App();
        TreeNode root = new TreeNode(8);
        root.left = new TreeNode(10);
        root.right = new TreeNode(6);
        root.left.left = new TreeNode(5);
        root.left.right = new TreeNode(7);
        root.right.left = new TreeNode(9);
        root.right.right = new TreeNode(11);
        System.out.println(app.Print(root));
    }

    @Test
    public void testMaxDepth() {
        App app = new App();
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        System.out.println(app.maxDepth(root)); // 3
    }

    @Test
    public void testHasPathSum() {

        App app = new App();
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        System.out.println(app.hasPathSum(root, 7)); // true
    }

    @Test
    public void testConvert() {
        App app = new App();
        TreeNode root = new TreeNode(10);
        root.left = new TreeNode(6);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(8);
        root.right = new TreeNode(14);
        root.right.left = new TreeNode(12);
        root.right.right = new TreeNode(16);

        TreeNode res = app.Convert(root);
        while (res != null) {
            System.out.println(res.val);
            res = res.right;
        }
    }

    @Test
    public void testIsSymmetrical() {
        App app = new App();
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(2);
        root.left.left = new TreeNode(3);
        root.left.right = new TreeNode(4);
        root.right.left = new TreeNode(4);
        root.right.right = new TreeNode(3);
        boolean symmetrical = app.isSymmetrical(root);
        System.out.println(symmetrical);
    }

    @Test
    public void testMergeTrees() {
        App app = new App();
        TreeNode root1 = new TreeNode(1);
        root1.left = new TreeNode(3);
        root1.right = new TreeNode(2);
        root1.left.left = new TreeNode(5);

        TreeNode root2 = new TreeNode(2);
        root2.left = new TreeNode(1);
        root2.right = new TreeNode(3);
        root2.left.right = new TreeNode(4);
        root2.right.right = new TreeNode(7);
        TreeNode res = app.mergeTrees(root1, root2);
        System.out.println(app.levelOrder(res));
    }

    @Test
    public void testMirror() {
        App app = new App();
        TreeNode root = new TreeNode(8);
        root.left = new TreeNode(6);
        root.right = new TreeNode(10);
        root.left.left = new TreeNode(5);
        root.left.right = new TreeNode(7);
        root.right.left = new TreeNode(9);
        root.right.right = new TreeNode(11);
        TreeNode res = app.Mirror(root);
        System.out.println(app.levelOrder(res));
    }

    @Test
    public void testIsValidBST_dfs() {
        App app = new App();
        /*TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);*/

        TreeNode root = new TreeNode(2);
        root.left = new TreeNode(1);
        root.right = new TreeNode(3);

        System.out.println(app.isValidBST(root));
    }

    @Test
    public void testIsCompleteTree() {
        App app = new App();
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        root.right.left = new TreeNode(6);
        boolean res = app.isCompleteTree(root);
        System.out.println(res);
    }

    @Test
    public void testIsBalanced_Solution() {
        App app = new App();
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        root.right.left = new TreeNode(6);
        root.right.right = new TreeNode(7);
        System.out.println(app.IsBalanced_Solution(root));
    }

    @Test
    public void testLowestCommonAncestor() {
        App app = new App();
        TreeNode root = new TreeNode(7);
        root.left = new TreeNode(1);
        root.right = new TreeNode(12);
        root.left.left = new TreeNode(0);
        root.left.right = new TreeNode(4);
        root.right.left = new TreeNode(11);
        root.right.right = new TreeNode(14);
        root.left.right.left = new TreeNode(3);
        root.left.right.right = new TreeNode(5);
        int res = app.lowestCommonAncestor(root, 1, 12);
        System.out.println(res);
    }

    @Test
    public void testLowestCommonAncestor2() {
        App app = new App();
        TreeNode root = new TreeNode(3);
        root.left = new TreeNode(5);
        root.right = new TreeNode(1);
        root.left.left = new TreeNode(6);
        root.left.right = new TreeNode(2);
        root.right.left = new TreeNode(0);
        root.right.right = new TreeNode(8);
        root.left.right.left = new TreeNode(7);
        root.left.right.right = new TreeNode(4);
        System.out.println(app.lowestCommonAncestor2(root, 5, 1));
    }

    @Test
    public void testSerialize() {
        App app = new App();
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.right.left = new TreeNode(6);
        root.right.right = new TreeNode(7);
        System.out.println(app.Serialize(root));
    }

    @Test
    public void testDeserialize() {
        App app = new App();
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(9);
        root.left.right = new TreeNode(5);
        root.right.left = new TreeNode(6);
        root.right.right = new TreeNode(7);
        TreeNode res = app.Deserialize(app.Serialize(root));
        System.out.println(app.levelOrder(res));
    }

    @Test
    public void testSolve() {
        App app = new App();
        // 前序  和  中序数组
        int[] pre = new int[]{1, 2, 4, 5, 3};
        int[] in = new int[]{4, 2, 5, 1, 3};
        System.out.println(Arrays.toString(app.solve(pre, in)));
    }

    @Test
    public void testRebuildTree() {
        App app = new App();
        TreeNode res = app.rebuildTree(new int[]{1, 2, 4, 5, 3}, new int[]{4, 2, 5, 1, 3});
        ArrayList<ArrayList<Integer>> result = app.levelOrder(res);
        System.out.println(result);
    }

    @Test
    public void testMyStackMin() {
        App.MyStackMin myStackMin = new App.MyStackMin();
        myStackMin.push(-1);
        myStackMin.push(2);
        int min = myStackMin.min();
        int top = myStackMin.top();
        System.out.println(min);
        System.out.println(top);
        myStackMin.pop();
        myStackMin.push(1);
        int top2 = myStackMin.top();
        int min2 = myStackMin.min();
        System.out.println(top2);
        System.out.println(min2);
    }

    @Test
    public void testIsValid() {
        App app = new App();

        System.out.println(app.isValid("{}[]()"));
        System.out.println(app.isValid("{]()"));
        System.out.println(app.isValid("{[()]}"));

    }

    @Test
    public void testMaxInWindows() {
        App app = new App();
        System.out.println(app.maxInWindows(new int[]{2, 3, 4, 2, 6, 2, 5, 1}, 3));
    }

    @Test
    public void testGetLeastNumbers_Solution() {
        App app = new App();
        System.out.println(app.GetLeastNumbers_Solution(new int[]{4, 5, 1, 6, 2, 7, 3, 8}, 4));
    }

    @Test
    public void testFindKth() {
        App app = new App();
        System.out.println(app.findKth(new int[]{10, 10, 9, 9, 8, 7, 5, 6, 4, 3, 4, 2}, 12, 3));
    }

    @Test
    public void testTwoSum() {
        App app = new App();
        System.out.println(Arrays.toString(app.twoSum(new int[]{3, 2, 4}, 6)));
    }

    @Test
    public void testMoreThanHalfNum_Solution() {
        App app = new App();
        System.out.println(app.MoreThanHalfNum_Solution(new int[]{1, 2, 3, 2, 2, 2, 5, 4, 2}));
    }

    @Test
    public void testFindNumsAppearOnce() {
        App app = new App();
        System.out.println(Arrays.toString(app.FindNumsAppearOnce(new int[]{1, 4, 1, 6})));
    }

    @Test
    public void testMinNumberDisappeared() {
        App app = new App();
        System.out.println(app.minNumberDisappeared(new int[]{-2, 3, 4, 1, 5}));
    }

    @Test
    public void testThreeSum() {
        App app = new App();
        ArrayList<ArrayList<Integer>> arrayLists = app.threeSum(new int[]{-2, 0, 1, 1, 2});
        System.out.println(arrayLists);
    }

    @Test
    public void testPermute() {
        App app = new App();
        System.out.println(app.permute(new int[]{1, 2, 3}));
    }

    @Test
    public void testPermuteUnique() {
        App app = new App();
        System.out.println(app.permuteUnique(new int[]{1, 1, 2}));
    }

    @Test
    public void testTestSolve() {
        App app = new App();
        System.out.println(app.solve2(new char[][]{{'1', '1', '0', '0', '0'},
                {'0', '1', '0', '1', '1'},
                {'0', '0', '0', '1', '1'},
                {'0', '0', '0', '0', '0'},
                {'0', '0', '1', '1', '1'}}));
    }

    @Test
    public void testPermutation_backtrack() {
        App app = new App();
        System.out.println(app.Permutation("aab"));
    }

    @Test
    public void testNqueen() {
        App app = new App();
        System.out.println(app.Nqueen(8));
    }

    @Test
    public void testGenerateParenthesis() {
        App app = new App();
        ArrayList<String> res = app.generateParenthesis(3);
        System.out.println(res);
    }

    @Test
    public void testTestSolve1() {
        App app = new App();
        System.out.println(app.solve(new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}));
    }

    @Test
    public void testJumpFloor() {
        App app = new App();
        System.out.println(app.jumpFloor(7));
    }

    @Test
    public void testMinCostClimbingStairs() {
        App app = new App();
        System.out.println(app.minCostClimbingStairs(new int[]{1, 100, 1, 1, 1, 90, 1, 1, 80, 1}));
    }

    @Test
    public void testLCS() {
        App app = new App();
        System.out.println(app.LCS("1A2C3D4B56", "B1D23A456A"));
    }

    @Test
    public void testLCS2() {
        App app = new App();
        System.out.println(app.LCS2("1AB2345CD", "12345EF"));
    }

    @Test
    public void testUniquePaths2_dfs() {
        App app = new App();
//        System.out.println(app.uniquePaths2(2, 2));
        System.out.println(app.uniquePaths(3, 3));
    }

    @Test
    public void testMinPathSum() {
        App app = new App();
        System.out.println(app.minPathSum(new int[][]{{1, 3, 5, 9}, {8, 1, 3, 4}, {5, 0, 6, 1}, {8, 8, 4, 0}}));
    }

    @Test
    public void testSolve2() {
        App app = new App();
        System.out.println(app.solve2("12"));
    }

    @Test
    public void testMinMoney() {
        App app = new App();
        System.out.println(app.minMoney(new int[]{5, 2, 3}, 20));
    }

    @Test
    public void testLIS() {
        App app = new App();
        System.out.println(app.LIS(new int[]{3, 5, 7, 1, 2, 4, 6, 3, 8, 9, 5, 6}));
    }

    @Test
    public void testFindGreatestSumOfSubArray() {
        App app = new App();
        System.out.println(app.FindGreatestSumOfSubArray(new int[]{1, -2, 3, 10, -4, 7, 2, -5}));
    }

    @Test
    public void testGetLongestPalindrome() {
        App app = new App();
        System.out.println(app.getLongestPalindrome("ababc"));
    }

    @Test
    public void testRestoreIpAddresses() {
        App app = new App();
        ArrayList<String> strings = app.restoreIpAddresses("25525522135");
        System.out.println(strings);
    }

    @Test
    public void testRob() {
        App app = new App();
        System.out.println(app.rob(new int[]{1, 2, 3, 4}));
    }

    @Test
    public void testRob2() {
        App app = new App();
        System.out.println(app.rob2(new int[]{1, 2, 3, 4}));
    }

    @Test
    public void testMaxProfit() {
        App app = new App();
        System.out.println(app.maxProfit(new int[]{8, 9, 2, 5, 4, 7, 1}));
    }

    @Test
    public void testMaxProfit2() {
        App app = new App();
        System.out.println(app.maxProfit2(new int[]{8, 9, 2, 5, 4, 7, 1}));
    }

    @Test
    public void testMaxProfit3() {
        App app = new App();
        System.out.println(app.maxProfit3(new int[]{8, 9, 3, 5, 1, 3}));
    }

    @Test
    public void testEditDistance() {
        App app = new App();
        System.out.println(app.editDistance("nowcoder", "new"));
    }

    @Test
    public void testTrans() {
        App app = new App();
        System.out.println(app.trans("          ", 10));
    }

    @Test
    public void testLongestCommonPrefix() {
        App app = new App();
//        System.out.println(app.longestCommonPrefix(new String[]{"abca", "abcg", "abca", "abcf", "abcc"}));
        System.out.println(app.longestCommonPrefix(new String[]{"cadrwrer", "cadrwes"}));
    }

    @Test
    public void testSolve3() {
        App app = new App();
        System.out.println(app.solve3("172.16.254.1"));
    }

    @Test
    public void testTestSolve2() {
        App app = new App();
        System.out.println(app.solve("1", "99"));
    }

    @Test
    public void testTestMerge() {
        App app = new App();
        ArrayList<Interval> root = new ArrayList<>();
        root.add(new Interval(1, 4));
        root.add(new Interval(0, 2));
        ArrayList<Interval> merge = app.merge(root);
        for (Interval interval : merge) {
            System.out.println(interval.start + "   " + interval.end);
        }
    }

    @Test
    public void testMaxLength() {
        App app = new App();
        System.out.println(app.maxLength(new int[]{2, 3, 4, 5}));
    }

    @Test
    public void testMaxArea() {
        App app = new App();
        System.out.println(app.maxArea(new int[]{1, 7, 3, 2, 4, 5, 8, 2, 7}));
    }

    @Test
    public void testMaxWater() {
        App app = new App();
        System.out.println(app.maxWater(new int[]{3, 1, 2, 5, 2, 4}));
    }

    @Test
    public void testSpiralOrder() {
        App app = new App();
//        System.out.println(app.minWindow("XDOYEZODEYXNZ", "XYZ"));
        System.out.println(app.minWindow("lhibsbrpxssyuibsdicrucaega", "ebsdslcacpib"));
    }

}

































