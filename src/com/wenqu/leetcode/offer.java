import java.util.*;
import java.math.BigInteger;
public class offer {

    /**
     * 剑指offer 20：斐波那契数列
     * 动态规划解法
     * @param n
     * @return
     */
    public int fib(int n) {
        if (n == 0){
            return 0;
        }

        if (n == 1){
            return 1;
        }

        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            dp[i] = (dp[i - 1] + dp[i - 2]) % 1000000007;
        }

        return dp[n];
    }


    /**
     * 剑指offer 21：青蛙跳台阶问题
     * 动态规划解法
     * @param n
     * @return
     */
    public int numWays(int n) {
        if (n == 0){
            return 1;
        }

        if (n == 1){
            return 1;
        }

        int[] dp = new int[n + 1];
        // 初始化
        dp[1] = 1;  // 第一个台阶只有一种跳法
        dp[2] = 2; // 第二个台阶存在两种跳法

        for (int i = 3; i <=n ; i++) {
            dp[i] = (dp[i - 1] + dp[i - 2]) % 1000000007; // 解释：该台阶的跳法就是前两个台阶跳法之和
        }

        return dp[n];
    }

    /**
     * 剑指offer 22：股票的最大利润
     * 动态规划解法
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        if (prices.length == 0) return 0;
        // 1、确定dp及其下标含义
        // dp[i][0] 表示第 i 天持有股票所得最多现金     dp[i][1] 表示第 i 不持有股票所得最多现金
        int[][] dp = new int[prices.length + 1][2];
        // 2、确定递推公式    dp[i][0] 分为第 i 天之前就买入 还是当天买入    dp[i][1]  分为第 i 天之前就卖出 还是当天卖出
        // dp[i][0] = fmax(dp[i - 1][0], -prices[i])     dp[i][1] = fmax(dp[i - 1][1], dp[i - 1][0] + prices[i])
        // 3、初始化dp    根据递推公式初始化 dp[0][0]   dp[0][1]
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        // 4、确定遍历顺序   i 状态依赖于 i - 1  ， 从前往后遍历
        for (int i = 1; i < prices.length; i++)
        {
            dp[i][0] = Math.max(dp[i - 1][0], -prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
        }
        // 5、举例推导出返回值
        return dp[prices.length - 1][1];
    }

    /**
     * 剑指offer 23：连续子数组的最大和
     * 动态规划解法
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
        int pre = 0, // 初始化前一个元素
        maxAns = nums[0]; // 将第一个元素初始化为 最大值
        for (int x : nums) {
            pre = Math.max(pre + x, x); // 跟新前一个元素
            maxAns = Math.max(maxAns, pre); // 跟新最大值
        }
        return maxAns;
    }

    /**
     * 剑指offer 24：礼物的最大价值
     * 动态规划解法
     * @param grid
     * @return
     */
    public int maxValue(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(i == 0 && j == 0) continue; // 第一个元素跳过
                if(i == 0) grid[i][j] += grid[i][j - 1] ; // 第一行元素
                else if(j == 0) grid[i][j] += grid[i - 1][j]; // 第一列元素
                else grid[i][j] += Math.max(grid[i][j - 1], grid[i - 1][j]);
            }
        }
        return grid[m - 1][n - 1];  // 右下角元素为最大元素
    }

    /**
     * 剑指offer 25：把数字翻译成字符串
     * 动态规划解法
     * 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。
     * 一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
     * @param num
     * @return
     */
    public int translateNum(int num) {
        String src = String.valueOf(num);  // 转化为字符串
        int p = 0, q = 0, r = 1;
        for (int i = 0; i < src.length(); ++i) { // 遍历字符串
            p = q;
            q = r;
            r = 0;
            r += q;
            if (i == 0) {
                continue;
            }
            String pre = src.substring(i - 1, i + 1);
            if (pre.compareTo("25") <= 0 && pre.compareTo("10") >= 0) {
                r += p;
            }
        }
        return r;

    }

    /**
     * 剑指offer 26：最长不含重复字符的子字符串
     * 动态规划解法
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> dic = new HashMap<>();
        int res = 0;
        int tmp = 0;
        for(int j = 0; j < s.length(); j++) {
            int i = j - 1;
            while(i >= 0 && s.charAt(i) != s.charAt(j)) i--; // 线性查找 i   不重复字符
            tmp = tmp < j - i ? tmp + 1 : j - i; // dp[j - 1] -> dp[j]
            res = Math.max(res, tmp); // max(dp[j - 1], dp[j])
        }
        return res;

    }
    public static class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }
    /**
     * 剑指offer 27：删除链表的节点
     * 双指针法
     * @param head
     * @param val
     * @return
     */
    public ListNode deleteNode(ListNode head, int val) {

        if (head == null) return head;
        if (head.val == val) return head.next;

        ListNode pre = null;
        ListNode cur = head;

        while (cur != null){

            if (cur.val == val){
                pre.next = cur.next;
                break;
            }
            pre = cur;
            cur = cur.next;
        }

        return head;
    }

    /**
     * 剑指offer 28：链表中倒数第 k 个节点
     * 双指针法
     * @param head
     * @param k
     * @return
     */
    public ListNode getKthFromEnd(ListNode head, int k) {

        // 思路：快慢指针
        ListNode pre = head;
        ListNode cur = head;

        for (int i = 0; i < k; i++) {
            if (cur != null){
                cur = cur.next;
            }
        }

        while (cur != null){
            pre = pre.next;
            cur = cur.next;
        }
        return pre;
    }

    /**
     * 剑指offer 33：翻转单词顺序
     * 双指针法
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        s = s.trim(); // 删除首尾空格
        int j = s.length() - 1; // 两个指针都指向最后一个元素
        int i = s.length() - 1;
        StringBuilder res = new StringBuilder();

        while (i >= 0){
            while(i >= 0 && s.charAt(i) != ' ') i--; // 搜索首个空格   从后向前查找单词
            res.append(s.substring(i + 1, j + 1) + " "); // 添加单词
            while(i >= 0 && s.charAt(i) == ' ') i--; // 跳过单词间空格
            j = i; // j 指向下个单词的尾字符
        }


        return res.toString().trim(); // 转化为字符串并返回

    }

    public void backTracking(TreeNode root, int target, int sum, List<List<Integer>> res, List<Integer> temp){
        if (root == null) return;

        temp.add(root.val);
        sum += root.val;
        if (sum == target && root.left == null && root.right == null){
            res.add(new ArrayList<>(temp));
        }

        if (root.left != null) backTracking(root.left, target, sum, res, temp);
        if (root.right != null) backTracking(root.right, target, sum, res, temp);

        sum -= root.val;
        temp.remove(temp.size() - 1);
    }

    /**
     * 剑指offer 34：回溯搜索
     * @param root
     * @param target
     * @return
     */
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();
        backTracking(root, target, 0, res, temp);

        return res;
    }

    public void backTracking2(Node node, List<Node> res){
        if (node == null) return;
        if (node.left != null) backTracking2(node.left, res);
        res.add(node);
        if (node.right != null) backTracking2(node.right, res);
    }

    /**
     * 剑指offer 35：回溯搜索
     * @param root
     * @return
     */
    public Node treeToDoublyList(Node root){
        if (root == null) return root;

        List<Node> res = new ArrayList<>();
        backTracking2(root, res);


        Node doublyListHead = new Node();
        doublyListHead = res.get(0);
        res.get(0).left = res.get(res.size() - 1);
        for (int i = 0; i < res.size() - 1; i++) {
            res.get(i).right = res.get(i + 1);
            res.get(i + 1).left = res.get(i);
        }
        res.get(res.size() - 1).right = res.get(0);

        return doublyListHead;
    }

    public void backTracking3(TreeNode node, List<TreeNode> res){
        if (node == null) return;
        if (node.left != null) backTracking3(node.left, res);
        res.add(node);
        if (node.right != null) backTracking3(node.right, res);
    }

    /**
     * 剑指offer 36：回溯搜索
     * @param root
     * @param k
     * @return
     */
    public int kthLargest(TreeNode root, int k){

        List<TreeNode> res = new ArrayList<>();
        backTracking3(root, res);
        return res.get(res.size() - k).val;
    }



    public void backTracking4(int[] nums, StringBuilder s, int count, StringBuilder minValue, int index){

        if (count == nums.length){

            // 比较字符串大小   （1）int转char，将数字加一个‘0’，并强制类型转换为char。
            //（2）char转int，将字符减一个‘0’即可。
            StringBuilder ss = new StringBuilder(s);
            for (int i = 0; i < ss.length(); i++) {
                if (((ss.charAt(i) - '0') < (minValue.charAt(i) - '0'))){
                    minValue = ss;
                    System.out.println(minValue);
                    break;
                }
            }
//            return;
        }

        for (int i = index; i < nums.length; i++) {
            int len = (nums[i] + "").length();
            s.append(nums[i]);
            count ++;
            backTracking4(nums, s, count, minValue, i + 1);
            s.delete(s.length() - len, s.length());
            count --;
        }

    }

    public void backTracking4(List<StringBuilder> res, StringBuilder temp, boolean[] used, int count, int[] nums){

        if (count == nums.length){
            res.add(new StringBuilder(temp));
            return;
        }

        for (int i = 0; i < nums.length; i++) {

            if (used[i]){
                continue;
            }

            temp.append(nums[i]);
            int len = (nums[i] + "").length();
            used[i] = true;
            count ++;
            backTracking4(res, temp, used, count, nums);
            temp.delete(temp.length() - len, temp.length());
            used[i] = false;
            count --;

        }


    }
    /**
     * 剑指offer 37：排序
     * 输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
     * 这道题思路是一道排列问题，使用回溯算法解决                 回溯解法超时了！！！！！
     * @param nums
     * @return
     */
    public String minNumber(int[] nums) {

        List<StringBuilder> res = new ArrayList<>();
        boolean[] used = new boolean[nums.length];

        backTracking4(res, new StringBuilder(), used, 0, nums);

        String minValue = String.valueOf(res.get(0));

        for (int i = 1; i < res.size(); i++) {
            for (int j = 0; j < res.get(0).length(); j++) {
                if ((minValue.charAt(j) - '0') < (res.get(i).charAt(j) - '0')){
                    break;
                }else if ((minValue.charAt(j) - '0') > (res.get(i).charAt(j) - '0')){
                    minValue = String.valueOf(res.get(i));
                    break;
                }
            }
        }

        return minValue;
    }



    /**
     * 剑指offer 38：排序
     * @param nums
     * @return
     */
    public boolean isStraight(int[] nums) {



        return false;
    }

    /**
     * 剑指offer 39：排序
     * @param arr
     * @param k
     * @return
     */
    public int[] getLeastNumbers(int[] arr, int k) {
        Arrays.sort(arr);
        int[] res = new int[k];
        System.arraycopy(arr, 0, res, 0, k);
        return res;
    }

    /**
     * 剑指offer 40：二叉树的深度
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        return (Math.max(left, right)) + 1;
    }

    public int getDeep(TreeNode root){

        if (root == null) return 0;

        int left = getDeep(root.left);
        int right = getDeep(root.right);

        return (Math.max(left, right)) + 1;

    }
    /**
     * 剑指offer 41：平衡二叉树
     * @param root
     * @return
     */
    public boolean isBalanced(TreeNode root) {
        if (root == null) return true;

        if (Math.abs(getDeep(root.left) - getDeep(root.right)) <= 1){
            return isBalanced(root.left) && isBalanced(root.right);
        }else {
            return false;
        }
    }

    /**
     * 剑指offer 42：求 1+2+。。。+n
     * @param n
     * @return
     */
    public int sumNums(int n) {
        boolean flag = n > 0 && (n += sumNums(n - 1)) > 0;
        return n;
    }

    /**
     * 剑指offer 43：二叉搜索树的最近公共祖先
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (p.val > root.val && q.val > root.val){
            return lowestCommonAncestor(root.right, p, q);
        }else if (p.val < root.val && q.val < root.val){
            return lowestCommonAncestor(root.left, p, q);
        }else {
            return root;
        }
    }


    /**
     * 剑指offer 44：二叉树的最近公共祖先
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q ){
            return root;
        }

        TreeNode left = lowestCommonAncestor2(root.left, p, q);
        TreeNode right = lowestCommonAncestor2(root.right, p, q);

        if (left == null && right == null) {
            return null;
        }
        if (left == null){
            return right;
        }
        if (right == null){
            return left;
        }
        return root;
    }


    /**
     * 剑指offer 45：重建二叉树  根据前序和中序遍历重建二叉树，这道题其实就是回溯思想  也具备分治思想
     * 首次开整分治算法
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {

        // 这里已经使用 java 实现 -> 需要写切片和找索引函数，这里使用 c++ 实现（迭代器比较方便）

        return null;
    }

    public double myPow1(double x, int n) {
        double res = 1;
        while (n > 0){
            res *= x;
            n --;
        }
        return res;

//        if (n <= 0) {
//            return 1;
//        }
//
//        return x * myPow(x, --n);
    }

    public double myPow2(double x, int n) {
        double res = 1;
        while (n < 0){
            res *= 1 / x;
            n ++;
        }
        return res;

//        if (n >= 0) {
//            return 1;
//        }
//
//        return 1 / x * myPow(x, ++n);

    }
    /**
     * 剑指offer 46：数值的整数次方       可以是正整数，也可以是负整数           超时！！！！！
     * 首次开整分治算法
     * @param x
     * @param n
     * @return
     */
    public double myPow(double x, int n) {
        if ((int)x == 1){
            return x;
        }
        if (n > 0) {
            return myPow1(x, n);
        }

        if (n < 0) {
            return myPow2(x, n);
        }

        return 1;
    }

    /**
     * 剑指offer 47：剪绳子2
     * 动态规划
     * @param n
     * @return
     */

    public int cuttingRope(int n) {

        // 因为 题目 n >= 2 所以不考虑，但在动规数组中需要作为初始化
//        if (n == 0 || n == 1){
//            return 0
//        }

        BigInteger dp[] = new BigInteger[n + 1];
        Arrays.fill(dp, BigInteger.valueOf(1));
        for (int i = 3; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                dp[i] = dp[i].max(BigInteger.valueOf(j * (i - j))).max(dp[i - j].multiply(BigInteger.valueOf(j)));
            }
        }
        return  dp[n].mod(BigInteger.valueOf(1000000007)).intValue();
    }


}


class Node {
    int val;
    Node left;
    Node right;
    Node() {}
    Node(int val) { this.val = val; }
    Node(int val, Node left, Node right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
