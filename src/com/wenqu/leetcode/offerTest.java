import org.junit.jupiter.api.Test;
import java.util.List;
class offerTest {

    @Test
    void fib() {
        offer offer = new offer();
        int n = 45;
        int res = offer.fib(n);
        System.out.println(res);  // 预期：134903163
    }

    @Test
    void numWays() {
        offer offer = new offer();
        int n = 100;

        int res = offer.numWays(n);
        System.out.println(res);

    }

    @Test
    void maxProfit() {
        offer offer = new offer();
//        int[] prices = new int[]{7, 1, 5, 3, 6, 4};
        int[] prices = new int[]{0, 2};

        int res = offer.maxProfit(prices);
        System.out.println(res);

    }

    @Test
    void maxSubArray() {
        offer offer = new offer();
        int[] nums = new int[]{-2,1,-3,4,-1,2,1,-5,4};
        int res = offer.maxSubArray(nums);
        System.out.println(res);
    }

    @Test
    void maxValue() {

        offer offer = new offer();
        int[][] grid = new int[][]{{1, 3, 1}, {1, 5, 1}, {4, 2, 1}};
        int res = offer.maxValue(grid);
        System.out.println(res);
    }

    @Test
    void translateNum() {

        offer offer = new offer();
        int num = 12258;
        int res = offer.translateNum(num);  // 5
        System.out.println(res);
    }

    @Test
    void lengthOfLongestSubstring() {

        offer offer = new offer();
        String s = "abcabcbb";
        int res = offer.lengthOfLongestSubstring(s);
        System.out.println(res);

    }

    @Test
    void deleteNode() {
        offer offer = new offer();
        offer.ListNode root = new offer.ListNode(-3);
        root.next = new offer.ListNode(5);
        root.next.next = new offer.ListNode(-99);
        int val = -3;
        offer.ListNode res = offer.deleteNode(root, val);

        while (res != null){
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    void getKthFromEnd() {
        offer offer = new offer();
        offer.ListNode root = new offer.ListNode(1);
        root.next = new offer.ListNode(2);
        root.next.next = new offer.ListNode(3);
        root.next.next.next = new offer.ListNode(4);
        root.next.next.next.next = new offer.ListNode(5);
        int k = 2;
        offer.ListNode res = offer.getKthFromEnd(root, k);

        while (res != null){
            System.out.println(res.val);
            res = res.next;
        }
    }

    @Test
    void reverseWords() {

        offer offer = new offer();
        String des = "   the sky is blue   ";
        String res = offer.reverseWords(des);
        System.out.println(res);
    }

    @Test
    public void testPathSum() {
//        输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
//        输出：[[5,4,11,2],[5,8,4,5]]
//        tree -> doublelist -> sort
//        no.k
        offer app = new offer();
        TreeNode root = new TreeNode(5);
        root.left = new TreeNode(4);
        root.right = new TreeNode(8);
        root.right.left = new TreeNode(13);
        root.right.right = new TreeNode(4);
        root.right.right.left = new TreeNode(5);
        root.right.right.right = new TreeNode(1);
        root.left.left = new TreeNode(11);
        root.left.left.left = new TreeNode(7);
        root.left.left.right = new TreeNode(2);

        int target = 26;
        List<List<Integer>> res = app.pathSum(root, target);

        System.out.println(res);

    }

    @Test
    public void testTreeToDoublyList() {
        offer app = new offer();
        Node root = new Node(4);
        root.left = new Node(2);
        root.left.left = new Node(1);
        root.left.right = new Node(3);
        root.right = new Node(5);
        Node res = app.treeToDoublyList(root);


    }
    @Test
    public void testKthLargest() {
        offer app = new offer();
        TreeNode root = new TreeNode(5);
        root.left = new TreeNode(3);
        root.right = new TreeNode(6);
        root.left.left = new TreeNode(2);
        root.left.right = new TreeNode(4);
        root.left.left.left = new TreeNode(1);
        int k = 3;
        int res = app.kthLargest(root, k);
        System.out.println(res); //4
    }

    @Test
    void minNumber() {

        offer offer = new offer();
        int[] des = new int[]{1, 2, 3, 5, 6, 7, 8, 9, 0};
        String res = offer.minNumber(des);
        System.out.println(res);

    }


    @Test
    void isStraight() {

    }

    @Test
    void getLeastNumbers() {

    }

    @Test
    void maxDepth() {

    }

    @Test
    void isBalanced() {

    }

    @Test
    void sumNums() {

    }

    @Test
    void lowestCommonAncestor() {

    }

    @Test
    void lowestCommonAncestor2() {

    }

    @Test
    void buildTree() {

        offer offer = new offer();
        int[] preorder = new int[]{};
        int[] inorder = new int[]{};
        TreeNode res = offer.buildTree(preorder, inorder);
        System.out.println(res.val);
    }

    @Test
    void myPow() {

        offer offer = new offer();
//        double x = 2.00000;
//        double x = 2.10000;
//        double x = 2.00000;
//        double x = 1.00000;
        double x = 2;
//        int n = 10;
//        int n = 3;
//        int n = -2;
//        int n = 2147483647;
        int n = -1;
        double res = offer.myPow(x, n);
        System.out.println(res);
    }

    @Test
    void cuttingRope() {
        offer offer = new offer();

        int res = offer.cuttingRope(10);
        System.out.println(res);
    }
}